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


def _grabcut_refine(bgr: np.ndarray, init_mask01: np.ndarray) -> np.ndarray:
    """
    Non-interactive GrabCut refinement.

    We seed GrabCut with:
    - probable foreground: init_mask01
    - probable background: elsewhere
    """
    h, w = bgr.shape[:2]
    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    mask[init_mask01 > 0] = cv2.GC_PR_FGD

    # Strong background prior on borders.
    border = max(5, int(round(min(h, w) * 0.03)))
    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, None, bgd_model, fgd_model, 4, mode=cv2.GC_INIT_WITH_MASK)
    out01 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    out01 = cv2.morphologyEx(out01, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    out01 = _largest_component(out01)
    return out01


def segment_glove(bgr: np.ndarray) -> SegmentationResult:
    """
    Produce a glove mask robust to moderate lighting/background changes.

    Returns:
    - glove_mask: raw mask (may contain holes)
    - glove_mask_filled: same mask with internal holes filled (useful for shape features)
    """
    h, w = bgr.shape[:2]
    cand_km = _kmeans_segment(bgr, k=3)
    cand_th = _threshold_segment_lab_l(bgr)

    # Always include a centered rectangle prior as last resort.
    rect = np.zeros((h, w), dtype=np.uint8)
    margin = int(round(min(h, w) * 0.08))
    rect[margin:-margin, margin:-margin] = 1

    cands = [
        ("kmeans", cand_km),
        ("labL_otsu", cand_th),
        ("rect", rect),
    ]
    best_method, best_init = max(cands, key=lambda kv: _score_mask(bgr, kv[1]))

    refined01 = _grabcut_refine(bgr, best_init)
    # Seal small gaps so interior holes (true defects) remain enclosed and detectable.
    refined01 = cv2.morphologyEx(refined01, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    filled01 = _fill_holes(refined01)

    return SegmentationResult(
        glove_mask=(refined01 * 255).astype(np.uint8),
        glove_mask_filled=(filled01 * 255).astype(np.uint8),
        method=f"{best_method}+grabcut",
    )
