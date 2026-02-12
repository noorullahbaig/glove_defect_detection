from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SegmentationResult:
    glove_mask: np.ndarray  # uint8 (0 or 255)
    glove_mask_filled: np.ndarray  # uint8 (0 or 255), with holes filled
    method: str


def _largest_component(mask01: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask01.astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask01
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = 1 + int(np.argmax(areas))
    return (labels == best_idx).astype(np.uint8)


def _fill_holes(mask01: np.ndarray) -> np.ndarray:
    """
    Fill holes in a binary mask by flood-filling the background from the border.
    """
    h, w = mask01.shape[:2]
    inv = (1 - mask01).astype(np.uint8) * 255
    flood = inv.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, seedPoint=(0, 0), newVal=0)
    holes = (flood > 0).astype(np.uint8)
    return np.clip(mask01 + holes, 0, 1).astype(np.uint8)


def _k(size: int) -> np.ndarray:
    size = max(1, int(size))
    return np.ones((size, size), np.uint8)


def _center_prior(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    nx = (xx - cx) / (0.44 * w + 1e-6)
    ny = (yy - cy) / (0.50 * h + 1e-6)
    dist2 = nx * nx + ny * ny
    prior = np.exp(-dist2).astype(np.float32)
    return prior


def _border_touch_ratio(mask01: np.ndarray) -> float:
    m = (mask01 > 0).astype(np.uint8)
    area = float(m.sum())
    if area <= 0:
        return 0.0
    h, w = m.shape[:2]
    band = max(5, int(round(min(h, w) * 0.03)))
    border = np.zeros_like(m, dtype=np.uint8)
    border[:band, :] = 1
    border[-band:, :] = 1
    border[:, :band] = 1
    border[:, -band:] = 1
    return float((m & border).sum()) / float(area + 1e-6)


def _disconnect_border_leaks(mask01: np.ndarray) -> np.ndarray:
    """
    Remove border-connected leakage by erode-reconstructing the most central body.
    """
    m = (mask01 > 0).astype(np.uint8)
    if m.sum() < 250:
        return m

    h, w = m.shape[:2]
    k = max(3, int(round(min(h, w) * 0.009)))
    er = cv2.erode(m, _k(k), iterations=1)
    if er.sum() < 120:
        return m

    num, labels, stats, cent = cv2.connectedComponentsWithStats(er, connectivity=8)
    if num <= 1:
        return m

    cy, cx = h / 2.0, w / 2.0
    best_idx = -1
    best_score = -1e9
    for i in range(1, num):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < 80:
            continue
        mx, my = float(cent[i, 0]), float(cent[i, 1])
        dist = ((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5
        center = 1.0 - dist / (((h * h + w * w) ** 0.5) + 1e-6)
        score = area * (0.7 + 0.7 * max(0.0, center))
        if score > best_score:
            best_score = score
            best_idx = i
    if best_idx < 1:
        return m

    core = (labels == best_idx).astype(np.uint8)
    recon = cv2.dilate(core, _k(k + 2), iterations=1)
    out = (m & recon).astype(np.uint8)
    if out.sum() < 0.45 * m.sum():
        return m
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, _k(max(3, k - 1)), iterations=1)
    out = _largest_component(out)
    return out


def _prune_boundary_color_outliers(bgr: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    """
    Remove likely background leakage on the boundary by checking LAB color
    consistency against an eroded interior region.
    """
    m = (mask01 > 0).astype(np.uint8)
    if m.sum() < 100:
        return m

    h, w = m.shape[:2]
    erode_sz = max(3, int(round(min(h, w) * 0.012)))
    core = cv2.erode(m, _k(erode_sz), iterations=1)
    if core.sum() < 80:
        return m

    boundary = (m - core).astype(np.uint8)
    if boundary.sum() < 20:
        return m

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    med = np.median(lab[core > 0], axis=0)
    de = np.linalg.norm(lab - med.reshape(1, 1, 3), axis=2)
    core_de = de[core > 0]
    if core_de.size == 0:
        return m
    thr = float(np.percentile(core_de, 93) + 5.0)
    thr = max(8.0, min(42.0, thr))

    core_ab = lab[:, :, 1:3][core > 0]
    med_ab = np.median(core_ab, axis=0)
    ab_dist = np.linalg.norm(lab[:, :, 1:3] - med_ab.reshape(1, 1, 2), axis=2)
    core_ab_dist = np.linalg.norm(core_ab - med_ab.reshape(1, 2), axis=1)
    ab_thr = float(np.percentile(core_ab_dist, 93) + 4.0) if core_ab_dist.size else 15.0
    ab_thr = max(4.0, min(26.0, ab_thr))

    keep_boundary = ((boundary > 0) & (de <= thr) & (ab_dist <= ab_thr)).astype(np.uint8)
    core_l = lab[:, :, 0][core > 0]
    if core_l.size and float(np.mean(core_l)) > 115.0:
        l_lo = float(np.percentile(core_l, 5) - 18.0)
        keep_boundary = (keep_boundary & (lab[:, :, 0] >= l_lo)).astype(np.uint8)

    out = (core | keep_boundary).astype(np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, _k(max(5, erode_sz + 2)), iterations=1)
    out = _largest_component(out)
    return out


def _select_component_centered(mask01: np.ndarray) -> np.ndarray:
    m = (mask01 > 0).astype(np.uint8)
    if m.sum() < 50:
        return m
    h, w = m.shape[:2]
    prior = _center_prior(h, w)
    num, labels, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m

    best_idx = -1
    best_score = -1e9
    for i in range(1, num):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < 120:
            continue
        comp = (labels == i).astype(np.uint8)
        cw = float((prior * comp).sum()) / float(area + 1e-6)
        bt = _border_touch_ratio(comp)
        score = area * (0.55 + 0.95 * cw) - (2200.0 * bt)
        if score > best_score:
            best_score = score
            best_idx = i
    if best_idx < 1:
        return _largest_component(m)
    return (labels == best_idx).astype(np.uint8)


def _border_model_segment(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment foreground by modeling border pixels as background in LAB space.

    Returns:
    - candidate foreground mask (0/1)
    - sure background mask (0/1) for GrabCut seeding
    """
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    prior = _center_prior(h, w)

    border_w = max(10, int(round(min(h, w) * 0.08)))
    border = np.zeros((h, w), dtype=np.uint8)
    border[:border_w, :] = 1
    border[-border_w:, :] = 1
    border[:, :border_w] = 1
    border[:, -border_w:] = 1
    border_px = lab[border > 0]
    if border_px.shape[0] < 80:
        return np.zeros((h, w), np.uint8), border

    med = np.median(border_px, axis=0)
    mad = np.median(np.abs(border_px - med.reshape(1, 3)), axis=0)
    mad = np.maximum(mad, np.array([3.0, 2.0, 2.0], dtype=np.float32))

    z = (lab - med.reshape(1, 1, 3)) / mad.reshape(1, 1, 3)
    dist = np.sqrt(np.sum(z * z, axis=2)).astype(np.float32)
    dist = cv2.GaussianBlur(dist, (5, 5), 0)

    d8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = (th > 0).astype(np.uint8)

    # Reinforce center-seeking foreground and suppress obvious border background.
    fg = (fg & (prior > 0.10)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _k(5), iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _k(9), iterations=2)
    fg = _select_component_centered(fg)

    border_dist = dist[border > 0]
    bg_thr = float(np.percentile(border_dist, 82)) if border_dist.size else float(np.percentile(dist, 35))
    sure_bg = ((dist <= bg_thr) & (prior < 0.72)).astype(np.uint8)
    sure_bg |= border.astype(np.uint8)

    if fg.sum() < int(0.015 * h * w):
        # fallback to a conservative center prior if background model was too strict.
        fg = ((prior > 0.38) & (d8 > int(np.percentile(d8, 55)))).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _k(5), iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _k(9), iterations=1)
        fg = _select_component_centered(fg)
    return fg.astype(np.uint8), sure_bg.astype(np.uint8)


def _kmeans_segment(bgr: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Segment using k-means in LAB space.
    Returns a 0/1 mask for the best-scoring component.
    """
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    # Use a,b for chromatic clustering; include L with a small weight.
    lab_f = lab.reshape(-1, 3).astype(np.float32)
    lab_f[:, 0] *= 0.5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
    _, labels, centers = cv2.kmeans(lab_f, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(h, w)

    # Score components by: area * centeredness (glove usually near center).
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    center_weight = 1.0 - (dist / (dist.max() + 1e-6))

    best_score = -1.0
    best_mask = np.zeros((h, w), np.uint8)
    for i in range(k):
        m = (labels == i).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
        m = _largest_component(m)
        area = float(m.sum())
        if area < 0.02 * h * w:
            continue
        cw = float((center_weight * m).sum()) / (area + 1e-6)
        score = area * (0.6 + 0.8 * cw)
        if score > best_score:
            best_score = score
            best_mask = m

    return best_mask


def _threshold_segment_lab_l(bgr: np.ndarray) -> np.ndarray:
    """
    Fallback segmentation for cases where the glove is significantly brighter
    than the background (common in lab/bench setups).
    """
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]
    l_blur = cv2.GaussianBlur(l, (5, 5), 0)
    _, th = cv2.threshold(l_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (th > 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
    mask = _largest_component(mask)
    # Reject trivial masks
    if mask.sum() < 0.02 * h * w:
        return np.zeros((h, w), np.uint8)
    return mask


def _score_mask(bgr: np.ndarray, mask01: np.ndarray) -> float:
    h, w = mask01.shape[:2]
    area = float(mask01.sum())
    if area <= 0:
        return -1e9

    # Reasonable area constraints.
    frac = area / float(h * w)
    if frac < 0.05 or frac > 0.90:
        return -1e9

    x, y, ww, hh = cv2.boundingRect(mask01.astype(np.uint8))
    extent = area / float(ww * hh + 1e-6)

    # Penalize masks that hug the border too much (often background leakage).
    border = max(5, int(round(min(h, w) * 0.04)))
    border_band = np.zeros_like(mask01, dtype=np.uint8)
    border_band[:border, :] = 1
    border_band[-border:, :] = 1
    border_band[:, :border] = 1
    border_band[:, -border:] = 1
    border_touch = float((mask01 & border_band).sum()) / (area + 1e-6)

    # Prefer masks whose mean brightness differs from outside (foreground/background separation).
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.float32)
    in_mean = float(l[mask01 > 0].mean()) if area > 0 else 0.0
    out_mean = float(l[mask01 == 0].mean()) if area < h * w else in_mean
    sep = (in_mean - out_mean) / 255.0

    # Centrality: glove tends to be near the center (soft prior).
    cy, cx = h / 2.0, w / 2.0
    m = cv2.moments(mask01.astype(np.uint8))
    if abs(m["m00"]) < 1e-6:
        center = 0.0
    else:
        mx = m["m10"] / m["m00"]
        my = m["m01"] / m["m00"]
        dist = ((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5
        center = 1.0 - dist / (((h * h + w * w) ** 0.5) + 1e-6)

    # Combine.
    score = 0.0
    score += 2.0 * extent
    score += 1.0 * center
    score += 1.2 * max(0.0, sep)  # only reward positive separation
    score -= 2.5 * border_touch
    score += 0.2 * min(1.0, frac / 0.35)
    return float(score)


def _grabcut_refine(
    bgr: np.ndarray,
    init_mask01: np.ndarray,
    aux_mask01: np.ndarray | None = None,
    sure_bg_mask01: np.ndarray | None = None,
) -> np.ndarray:
    """
    Non-interactive GrabCut refinement.

    We seed GrabCut with:
    - probable foreground: init_mask01
    - probable background: elsewhere
    """
    h, w = bgr.shape[:2]
    init = (init_mask01 > 0).astype(np.uint8)
    aux = init if aux_mask01 is None else (aux_mask01 > 0).astype(np.uint8)

    union = ((init > 0) | (aux > 0)).astype(np.uint8)
    inter = ((init > 0) & (aux > 0)).astype(np.uint8)
    if inter.sum() < 0.004 * h * w:
        inter = init.copy()

    erode_sz = max(3, int(round(min(h, w) * 0.012)))
    dilate_sz = max(5, int(round(min(h, w) * 0.02)))
    sure_fg = cv2.erode(inter, _k(erode_sz), iterations=1)
    prob_fg = cv2.dilate(union, _k(dilate_sz), iterations=1)
    sure_bg = (1 - cv2.dilate(prob_fg, _k(max(11, int(round(min(h, w) * 0.09)))), iterations=1)).astype(np.uint8)

    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    mask[prob_fg > 0] = cv2.GC_PR_FGD
    mask[sure_fg > 0] = cv2.GC_FGD
    mask[sure_bg > 0] = cv2.GC_BGD
    if sure_bg_mask01 is not None:
        mask[(sure_bg_mask01 > 0)] = cv2.GC_BGD

    # Strong background prior on borders.
    border = max(5, int(round(min(h, w) * 0.03)))
    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, None, bgd_model, fgd_model, 5, mode=cv2.GC_INIT_WITH_MASK)
    out01 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    # Cap expansion from the seed to reduce dark-background bleeding.
    expand_cap = cv2.dilate(union, _k(max(9, int(round(min(h, w) * 0.06)))), iterations=1)
    out01 &= expand_cap

    out01 = cv2.morphologyEx(out01, cv2.MORPH_OPEN, _k(5), iterations=1)
    out01 = cv2.morphologyEx(out01, cv2.MORPH_CLOSE, _k(7), iterations=1)
    out01 = _select_component_centered(out01)
    out01 = _prune_boundary_color_outliers(bgr, out01)
    bt = _border_touch_ratio(out01)
    if bt > 0.11:
        trimmed = _disconnect_border_leaks(out01)
        if _border_touch_ratio(trimmed) + 0.02 < bt:
            out01 = trimmed
    return out01


def segment_glove(bgr: np.ndarray) -> SegmentationResult:
    """
    Produce a glove mask robust to moderate lighting/background changes.

    Returns:
    - glove_mask: raw mask (may contain holes)
    - glove_mask_filled: same mask with internal holes filled (useful for shape features)
    """
    h, w = bgr.shape[:2]
    cand_bg, sure_bg = _border_model_segment(bgr)
    cand_km = _kmeans_segment(bgr, k=3)
    cand_th = _threshold_segment_lab_l(bgr)

    # Always include a centered rectangle prior as last resort.
    rect = np.zeros((h, w), dtype=np.uint8)
    margin = int(round(min(h, w) * 0.08))
    rect[margin:-margin, margin:-margin] = 1

    base = [
        ("border_model", cand_bg),
        ("kmeans", cand_km),
        ("labL_otsu", cand_th),
    ]
    scored_base = sorted([(m, mk, _score_mask(bgr, mk)) for m, mk in base], key=lambda x: float(x[2]), reverse=True)
    best_method, best_init, best_score = scored_base[0]
    aux_init = scored_base[1][1] if len(scored_base) > 1 else best_init

    # Use centered rectangle only as last resort; don't let it outscore real candidates.
    if float(best_score) < -1e8:
        best_method = "rect"
        best_init = rect
        aux_init = rect

    refined01 = _grabcut_refine(bgr, best_init, aux_mask01=aux_init, sure_bg_mask01=sure_bg)
    # Seal small gaps so interior holes (true defects) remain enclosed and detectable.
    refined01 = cv2.morphologyEx(refined01, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    filled01 = _fill_holes(refined01)

    return SegmentationResult(
        glove_mask=(refined01 * 255).astype(np.uint8),
        glove_mask_filled=(filled01 * 255).astype(np.uint8),
        method=f"{best_method}+grabcut",
    )
