from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SegmentationResult:
    glove_mask: np.ndarray  # uint8 (0 or 255)
    glove_mask_filled: np.ndarray  # uint8 (0 or 255), with holes filled
    method: str


@dataclass(frozen=True)
class SegmentationConfig:
    """
    Optional knobs for tuning segmentation strictness (used by the Streamlit UI).
    """

    force_candidate: str | None = None  # {"bg_dist","kmeans","labL_hi","labL_lo","edge_closed","chroma_fg","texture_fg"}
    force_refine: str | None = None  # {"grabcut","watershed"}
    profile_name: str = "balanced"
    gap_carve_mode: str = "hybrid"  # {"evidence","geometry","hybrid"}

    # Candidate validity gates (primarily for low-chroma/near-white gloves).
    chroma_min_area_frac: float = 0.20
    chroma_min_iou_silhouette: float = 0.25

    grabcut_cap_border_touch: float = 0.10
    grabcut_open_k: int = 3
    grabcut_close_k: int = 7

    edge_recover_enabled: bool = True
    edge_recover_kd_scale: float = 1.0
    edge_recover_ring_scale: float = 1.0
    edge_recover_max_area_mul: float = 1.22

    upright_enabled: bool = True
    upright_roi_frac: float = 0.60
    upright_min_improve_score: float = 0.04
    upright_min_improve_bg_like: float = 0.010

    upright_kd_scale: float = 1.8
    upright_ring_scale: float = 1.55
    upright_max_area_mul: float = 1.30

    valley_carve_enabled: bool = True
    valley_prom_frac: float = 0.05
    valley_notch_w_frac: float = 0.07
    valley_notch_depth_frac: float = 0.22
    valley_notch_depth_cap_frac: float = 0.42
    valley_max_count: int = 5

    bg_like_percentile: float = 96.0
    bg_like_margin: float = 6.0
    thin_dt_frac: float = 0.025

    # Shadow/webbing handling
    shadow_strictness: float = 1.0  # higher => stricter shadow classification (fewer pixels flagged)
    webbing_restore_enabled: bool = True
    webbing_trigger_ratio: float = 0.12  # trigger when (shadow|bg-like) ratio in finger boundary band exceeds this
    webbing_erode_k_frac: float = 0.018  # erosion kernel as fraction of min(h,w) in upright frame
    webbing_max_iters: int = 140
    webbing_convexity_enabled: bool = True
    webbing_solidity_thr: float = 0.92  # if area/hull_area exceeds this, silhouette is too convex
    webbing_depth_frac: float = 0.030  # min defect depth as fraction of min(bbox)
    webbing_max_defects: int = 2
    webbing_wedge_dilate_px: int = 7  # expand carved wedge slightly
    webbing_reach_overlap_thr: float = 0.18  # require this fraction of wedge to be border-reachable bg/shadow

    # Edge-reachable background carve (for shadows and thumb-index webbing gaps).
    # Finds background regions reachable from the border when treating strong edges as barriers,
    # then removes those pixels if they are mistakenly included near the mask boundary.
    reach_carve_enabled: bool = True
    reach_carve_strength: float = 1.0  # 0 disables; higher carves more aggressively
    reach_carve_max_rm_frac: float = 0.08  # max fraction of mask area removed per pass

    # Boundary pruning controls.
    boundary_prune_enabled: bool = True
    boundary_prune_edge_rescue: bool = True
    boundary_prune_skip_if_low_leak: bool = True

    # Gap carve safety controls.
    gap_geometry_fallback_solidity: float = 0.90
    gap_max_remove_frac: float = 0.06

    # Global post-cleanup for residual boundary halos on white backgrounds.
    global_halo_peel_enabled: bool = True
    global_halo_peel_trigger_ratio: float = 0.14
    global_halo_peel_max_remove_frac: float = 0.10
    global_halo_peel_dt_frac: float = 0.030

    # Conservative post-expand to recover clipped fingertips (mostly leather).
    tip_restore_enabled: bool = True
    tip_restore_max_add_frac: float = 0.015

    # Fingertip protection: prevents downstream refinement/carving from trimming
    # narrow finger tips that are already present in a good candidate.
    # Implemented as a soft "keep" band on the likely fingertip end of the mask.
    tip_protect_enabled: bool = True
    tip_protect_frac: float = 0.18  # fraction of glove length to protect at the fingertip end
    tip_protect_dilate_frac: float = 0.006  # small dilation of protected pixels (fraction of min(h,w))

    # Upright variant acceptance guard: refuse an upright refinement that discards
    # too much of the current fingertip band (helps stop cascaded tip clipping).
    upright_min_tip_retain: float = 0.92

    # ── Type-aware segmentation strategy ────────────────────────────────
    # Per-candidate score adjustments: {candidate_name: bonus} (e.g. {"edge_closed": 0.12}).
    candidate_boost: tuple = ()   # tuple of (name, bonus) pairs — frozen-safe
    candidate_suppress: tuple = ()  # tuple of (name, penalty) pairs

    # Type-specific candidate enables.
    latex_edge_multiscale: bool = False  # multi-scale Canny edge-closed candidate
    leather_saturation_candidate: bool = False  # HSV saturation-based candidate
    fabric_variance_candidate: bool = False  # local variance texture candidate

    # GrabCut seeding geometry (fractions of min(h,w)).
    grabcut_erode_frac: float = 0.012  # sure_fg erosion — smaller = more tip pixels in sure_fg
    grabcut_dilate_frac: float = 0.020  # prob_fg dilation — larger = more tip pixels as prob_fg
    grabcut_sure_bg_dilate_frac: float = 0.09  # sure_bg outer shell
    grabcut_inter_min_ratio: float = 0.78  # if init&aux overlap is too small, fall back to init for sure_fg
    # Post-GrabCut candidate-based cap: limit GrabCut expansion to dilated candidate.
    # 0.0 = disabled; positive = fraction of min(h,w) for candidate dilation.
    grabcut_candidate_cap_frac: float = 0.0

    # GrabCut tri-map seeding (multi-level Otsu on background-distance).
    grabcut_trimap_enabled: bool = False
    grabcut_trimap_classes: int = 3
    grabcut_trimap_only_if_low_confidence: bool = True
    grabcut_trimap_low_conf_bg_sep: float = 22.0
    grabcut_trimap_low_conf_edge_align: float = 0.10

    # Optional Chan–Vese candidate (region-based; useful for weak edges).
    chan_vese_enabled: bool = False
    chan_vese_max_side: int = 512
    chan_vese_iters: int = 80
    chan_vese_smoothing: int = 2
    chan_vese_only_if_low_confidence: bool = True
    chan_vese_low_conf_bg_sep: float = 22.0
    chan_vese_low_conf_edge_align: float = 0.10

    # Shape plausibility tie-breaker in candidate scoring.
    shape_plausibility_enabled: bool = True
    shape_plausibility_weight: float = 0.08

    # Edge-based finger separation: uses Canny edges to carve inter-finger
    # gaps and recover clipped fingertips.
    edge_finger_separation_enabled: bool = False


@dataclass(frozen=True)
class SegmentationDebug:
    """
    Optional debug payload for segmentation auditing/visualization.

    All masks are 0/1 uint8 unless noted otherwise.
    """

    bg_is_white: bool
    bg_is_uniform: bool
    bg_lab_median: tuple[float, float, float]
    bg_lab_mad: tuple[float, float, float]
    d_bg_u8: np.ndarray  # uint8 0..255
    grad_u8: np.ndarray  # uint8 0..255
    edges_u8: np.ndarray  # uint8 0 or 255
    candidates: dict[str, np.ndarray]  # method -> mask01
    scored: list[dict]  # list of {method, score, metrics...}
    candidate_validity: dict[str, dict[str, float | bool]]
    carve_stats: dict[str, float | str]
    prune_stats: dict[str, float | bool | str]
    grabcut_trimap_use: bool
    grabcut_trimap: dict[str, float | int | bool | str]
    chosen_method: str


@dataclass(frozen=True)
class _SegSignals:
    lab_u8: np.ndarray
    bg_med: np.ndarray  # (3,) float32
    bg_mad: np.ndarray  # (3,) float32
    border_mask01: np.ndarray  # 0/1
    center_prior: np.ndarray  # float32 0..1
    d_bg: np.ndarray  # float32
    d_bg_u8: np.ndarray  # uint8 0..255
    grad_u8: np.ndarray  # uint8 0..255
    edges01: np.ndarray  # 0/1
    edges_u8: np.ndarray  # 0/255
    bg_is_uniform: bool
    bg_is_white: bool


def _border_connected(mask01: np.ndarray, border01: np.ndarray) -> np.ndarray:
    """
    Return only the connected components in mask01 that touch border01.
    Both inputs are 0/1 uint8.
    """
    m = (mask01 > 0).astype(np.uint8)
    if m.sum() < 50:
        return np.zeros_like(m, dtype=np.uint8)
    num, labels = cv2.connectedComponents(m, connectivity=8)
    if num <= 1:
        return m
    sel = (border01 > 0) & (m > 0)
    if not np.any(sel):
        return np.zeros_like(m, dtype=np.uint8)
    labs = np.unique(labels[sel])
    labs = labs[labs != 0]
    if labs.size == 0:
        return np.zeros_like(m, dtype=np.uint8)
    return (np.isin(labels, labs) & (m > 0)).astype(np.uint8)


def _edge_reachable_map(sig: _SegSignals, *, strength: float) -> np.ndarray:
    """
    Border-reachable region using an edge barrier that targets the *object boundary*.

    We suppress dense internal texture edges by retaining only edge pixels that have
    both a nearby background-like side and a nearby foreground-like side (in d_bg space).

    Returns a 0/1 mask of pixels reachable from the image border without crossing the barrier.
    """
    h, w = sig.d_bg_u8.shape[:2]
    min_hw = float(min(h, w))
    s = float(max(0.0, min(2.5, strength)))

    d8 = sig.d_bg_u8.astype(np.float32)
    border_vals = d8[sig.border_mask01 > 0]
    if border_vals.size < 80:
        return sig.border_mask01.astype(np.uint8)

    thr_bg = float(np.percentile(border_vals, 98.5) + 6.0)
    thr_bg = float(max(6.0, min(90.0, thr_bg)))
    thr_fg = float(min(254.0, thr_bg + 38.0 + 10.0 * s))

    low = (d8 <= thr_bg).astype(np.uint8)
    high = (d8 >= thr_fg).astype(np.uint8)
    k_nb = int(round(5 + 2 * s))
    k_nb = int(min(13, max(5, k_nb | 1)))
    near_low = cv2.dilate(low, _k(k_nb), iterations=1)
    near_high = cv2.dilate(high, _k(k_nb), iterations=1)

    boundary_edges = (sig.edges01.astype(np.uint8) & (near_low > 0) & (near_high > 0)).astype(np.uint8)
    if boundary_edges.sum() < 80:
        return sig.border_mask01.astype(np.uint8)

    edge_dilate = int(round(3 + 4 * s))
    edge_dilate = int(min(19, max(3, edge_dilate | 1)))
    barrier = cv2.dilate(boundary_edges, _k(edge_dilate), iterations=1)
    passable = (1 - (barrier > 0).astype(np.uint8)).astype(np.uint8)

    reachable = _border_connected(passable, sig.border_mask01.astype(np.uint8))
    if reachable.sum() < 200:
        return sig.border_mask01.astype(np.uint8)

    # Reliability guard: reachable should mostly be background-like (low d_bg).
    vals = d8[reachable > 0]
    if vals.size:
        hi_frac = float(np.mean((vals >= thr_fg).astype(np.float32)))
        if hi_frac > 0.10:
            return sig.border_mask01.astype(np.uint8)

    return reachable.astype(np.uint8)


def _edge_reachable_bg_carve(
    mask01: np.ndarray,
    sig: _SegSignals,
    *,
    roi01: np.ndarray,
    cfg: SegmentationConfig,
) -> np.ndarray:
    """
    Carve out background/shadow pockets that are connected to the image border.

    Key idea: build an "edge barrier" (dilated edges + sure-foreground core), compute the
    set of pixels that are reachable from the border without crossing the barrier, and
    remove any such reachable pixels that are (incorrectly) inside the glove mask near
    its boundary (optionally limited to a ROI).
    """
    m = (mask01 > 0).astype(np.uint8)
    if m.sum() < 400:
        return m
    if not bool(cfg.reach_carve_enabled) or float(cfg.reach_carve_strength) <= 0.0:
        return m

    h, w = m.shape[:2]
    min_hw = float(min(h, w))
    strength = float(cfg.reach_carve_strength)

    # Combine chroma-based reachable background with an edge-barrier reachability.
    # Edge reachability helps for reflections (latex) where background pixels are not strictly bg-like.
    reach_bg = _reachable_background_mask(sig, strictness=float(cfg.shadow_strictness), cfg=cfg, roi01=roi01)
    reach_edge = _edge_reachable_map(sig, strength=float(strength))
    reachable = ((reach_bg > 0) | (reach_edge > 0)).astype(np.uint8)
    if reachable.sum() < 500:
        return m

    # Only carve near the boundary to avoid accidentally punching into the palm.
    band_k = int(round((0.018 + 0.010 * strength) * min_hw))
    band_k = int(min(31, max(5, band_k | 1)))
    band = (m & (1 - cv2.erode(m, _k(band_k), iterations=1))).astype(np.uint8)

    # Remove only where it's reachable background inside the mask (typical of shadows and the thumb-index gap).
    rm = (reachable > 0) & (m > 0) & (band > 0) & (roi01 > 0)
    rm_area = float(np.sum(rm))
    if rm_area < 30:
        return m

    max_rm = float(cfg.reach_carve_max_rm_frac) * float(m.sum() + 1e-6)
    if rm_area > max_rm:
        return m

    out = m.copy()
    out[rm] = 0
    out = _select_component_centered(out)
    return out.astype(np.uint8)


def _rotate_about_center(
    img: np.ndarray,
    deg: float,
    interp: int,
    *,
    border_value: tuple[int, int, int] | int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(deg), 1.0)
    if img.ndim == 2:
        out = cv2.warpAffine(img, m, (w, h), flags=int(interp), borderMode=cv2.BORDER_CONSTANT, borderValue=int(border_value))  # type: ignore[arg-type]
    else:
        out = cv2.warpAffine(img, m, (w, h), flags=int(interp), borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)  # type: ignore[arg-type]
    return out, m


def _pca_upright_deg(mask01: np.ndarray) -> float:
    ys, xs = np.where(mask01 > 0)
    if xs.size < 200:
        return 0.0
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    mean = pts.mean(axis=0, keepdims=True)
    pts0 = pts - mean
    cov = (pts0.T @ pts0) / float(max(1, pts0.shape[0] - 1))
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, int(np.argmax(vals))]  # principal axis in x,y
    theta = float(np.degrees(np.arctan2(v[1], v[0])))  # angle from +x
    # Image coordinates have +y downward; OpenCV positive angles rotate counter-clockwise
    # in this coordinate system. Due to the y-down convention and eigenvector sign
    # ambiguity, choose between ±(theta-90) by whichever yields a "tall" upright bbox.
    deg_a = float(90.0 - theta)
    deg_b = float(theta - 90.0)

    m0 = (mask01 > 0).astype(np.uint8)
    ra, _ = _rotate_about_center((m0 * 255).astype(np.uint8), deg_a, cv2.INTER_NEAREST, border_value=0)
    rb, _ = _rotate_about_center((m0 * 255).astype(np.uint8), deg_b, cv2.INTER_NEAREST, border_value=0)
    ma = (ra > 0).astype(np.uint8)
    mb = (rb > 0).astype(np.uint8)
    _, _, wa, ha = cv2.boundingRect(ma.astype(np.uint8))
    _, _, wb, hb = cv2.boundingRect(mb.astype(np.uint8))
    # Prefer the rotation that makes the glove more vertical (smaller w/h).
    ar_a = float(wa) / float(ha + 1e-6)
    ar_b = float(wb) / float(hb + 1e-6)
    return float(deg_a if ar_a <= ar_b else deg_b)


def _cuff_is_on_top(upright_mask01: np.ndarray) -> bool:
    m = (upright_mask01 > 0).astype(np.uint8)
    if m.sum() < 400:
        return False
    x, y, ww, hh = cv2.boundingRect(m.astype(np.uint8))
    if hh < 60 or ww < 60:
        return False
    band = max(12, int(round(0.18 * hh)))
    top = m[y : y + band, x : x + ww]
    bot = m[y + hh - band : y + hh, x : x + ww]

    def _median_row_width(block: np.ndarray) -> float:
        widths = (block > 0).sum(axis=1).astype(np.float32)
        widths = widths[widths > 0]
        return float(np.median(widths)) if widths.size else 0.0

    w_top = _median_row_width(top)
    w_bot = _median_row_width(bot)
    return bool(w_top > 1.10 * w_bot and w_top > 25.0)


def _fingers_on_top(mask01: np.ndarray) -> bool:
    """
    Decide whether the finger side is at the top of an upright (PCA-rotated) glove.

    We compare "valley" evidence on the top edge vs bottom edge:
    between fingers (and thumb-index gap) the silhouette has deeper indentations.
    """
    m = (mask01 > 0).astype(np.uint8)
    if m.sum() < 450:
        return True
    x, y, ww, hh = cv2.boundingRect(m.astype(np.uint8))
    if ww < 70 or hh < 110:
        return True

    def _edge_profile(end: str) -> tuple[int, float]:
        # Work within ~60% of bbox height from the chosen end.
        span = int(round(0.60 * float(hh)))
        span = int(max(40, min(int(hh), span)))
        prof = np.zeros((ww,), dtype=np.float32)
        valid = np.zeros((ww,), dtype=np.uint8)

        if end == "top":
            y0, y1 = y, y + span
            for xi in range(x, x + ww):
                col = m[y0:y1, xi]
                ys = np.where(col > 0)[0]
                if ys.size:
                    prof[xi - x] = float(ys[0])  # distance from top edge
                    valid[xi - x] = 1
        else:
            y0, y1 = y + hh - span, y + hh
            for xi in range(x, x + ww):
                col = m[y0:y1, xi]
                ys = np.where(col > 0)[0]
                if ys.size:
                    prof[xi - x] = float((span - 1) - int(ys[-1]))  # distance from bottom edge
                    valid[xi - x] = 1

        if int(valid.sum()) < max(40, int(0.30 * ww)):
            return 0, 0.0

        # Smooth profile to suppress knit texture noise.
        win = max(11, int(round(ww * 0.08)))
        if win % 2 == 0:
            win += 1
        win = min(win, 71)
        k = np.ones((win,), np.float32) / float(win)
        prof_s = np.convolve(prof, k, mode="same").astype(np.float32)

        # Valleys correspond to local maxima of distance-from-edge profile.
        prom_thr = float(max(7.0, 0.030 * float(hh)))
        valleys = 0
        for i in range(4, ww - 4):
            if valid[i] == 0:
                continue
            if not (prof_s[i - 1] < prof_s[i] > prof_s[i + 1]):
                continue
            left = float(np.min(prof_s[max(0, i - 10) : i]))
            right = float(np.min(prof_s[i + 1 : min(ww, i + 11)]))
            prom = float(prof_s[i] - max(left, right))
            if prom >= prom_thr:
                valleys += 1

        std = float(np.std(prof_s[valid > 0]))
        return int(valleys), float(std)

    vt, st = _edge_profile("top")
    vb, sb = _edge_profile("bottom")

    # Prefer the side with more valleys; tie-break by higher variability.
    if vt > vb:
        return True
    if vb > vt:
        return False
    return bool(st >= sb)


def _upright_normalize(bgr: np.ndarray, mask01: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate image + mask so the glove principal axis is vertical and fingers are on top.
    Returns (rot_bgr, rot_mask01, rot_matrix).
    """
    m0 = (mask01 > 0).astype(np.uint8)
    deg = _pca_upright_deg(m0)
    rot_bgr, mat = _rotate_about_center(bgr, deg, cv2.INTER_LINEAR, border_value=(255, 255, 255))
    rot_m_u8, _ = _rotate_about_center((m0 * 255).astype(np.uint8), deg, cv2.INTER_NEAREST, border_value=0)
    rot_m01 = (rot_m_u8 > 0).astype(np.uint8)

    # Ensure fingers are on top for downstream finger-ROI logic.
    # If ambiguous, fall back to the cuff-width heuristic.
    flip = not _fingers_on_top(rot_m01)
    if not flip and _cuff_is_on_top(rot_m01):
        flip = True
    if flip:
        rot_bgr, mat2 = _rotate_about_center(rot_bgr, 180.0, cv2.INTER_LINEAR, border_value=(255, 255, 255))
        rot_m_u8, _ = _rotate_about_center((rot_m01 * 255).astype(np.uint8), 180.0, cv2.INTER_NEAREST, border_value=0)
        rot_m01 = (rot_m_u8 > 0).astype(np.uint8)
        mat = np.vstack([mat, [0, 0, 1]])
        mat2 = np.vstack([mat2, [0, 0, 1]])
        mat = (mat2 @ mat)[:2, :]

    return rot_bgr, rot_m01, mat


def _finger_roi_from_mask(upright_mask01: np.ndarray, frac: float = 0.60) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    m = (upright_mask01 > 0).astype(np.uint8)
    x, y, ww, hh = cv2.boundingRect(m.astype(np.uint8))
    if ww <= 0 or hh <= 0:
        return np.zeros_like(m, dtype=np.uint8), (0, 0, 0, 0)
    roi = np.zeros_like(m, dtype=np.uint8)
    y_end = int(round(y + float(frac) * float(hh)))
    roi[y:y_end, x : x + ww] = 1
    return roi, (x, y, ww, hh)


def _bg_like_thr_u8(sig: _SegSignals) -> float:
    border_vals = sig.d_bg_u8[sig.border_mask01 > 0]
    if border_vals.size >= 80:
        # background-like pixels have *low* d_bg; allow carving slightly above typical background.
        thr = float(np.percentile(border_vals, 96) + 6.0)
    else:
        thr = float(np.percentile(sig.d_bg_u8, 20) + 18.0)
    return float(max(6.0, min(70.0, thr)))


def _bg_like_inside_ratio(mask01: np.ndarray, sig: _SegSignals, roi01: np.ndarray) -> float:
    m = (mask01 > 0).astype(np.uint8)
    if m.sum() < 200:
        return 0.0
    thr = _bg_like_thr_u8(sig)
    sel = (roi01 > 0) & (m > 0)
    denom = float(np.sum(sel))
    if denom < 100:
        return 0.0
    bg_like = (sig.d_bg_u8.astype(np.float32) <= float(thr))
    return float(np.sum(bg_like & sel)) / float(denom + 1e-6)


def _global_boundary_halo_peel(
    mask01: np.ndarray,
    sig: _SegSignals,
    cfg: SegmentationConfig,
    runtime: dict[str, object] | None = None,
) -> np.ndarray:
    m = (mask01 > 0).astype(np.uint8)
    if m.sum() < 500 or not bool(cfg.global_halo_peel_enabled):
        return m
    h, w = m.shape[:2]
    core_k = max(5, int(round(0.012 * min(h, w))))
    core = cv2.erode(m, _k(core_k), iterations=1)
    boundary = (m & (1 - core)).astype(np.uint8)
    if boundary.sum() < 120:
        return m

    border_vals = sig.d_bg_u8[sig.border_mask01 > 0]
    if border_vals.size >= 80:
        bg_like_thr = float(np.percentile(border_vals, float(cfg.bg_like_percentile)) + float(cfg.bg_like_margin))
        bg_like_thr = float(max(6.0, min(80.0, bg_like_thr)))
    else:
        bg_like_thr = _bg_like_thr_u8(sig)
    bg_like = (sig.d_bg_u8.astype(np.float32) <= float(bg_like_thr)).astype(np.uint8)
    shadow = _shadow_like_mask_anywhere(sig, strictness=float(cfg.shadow_strictness))
    reach_bg = _reachable_background_mask(sig, strictness=float(cfg.shadow_strictness), cfg=cfg, roi01=None)

    dt = cv2.distanceTransform((m * 255).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    thin_thr = float(max(7.0, float(cfg.global_halo_peel_dt_frac) * float(min(h, w))))
    thin = (dt <= thin_thr).astype(np.uint8)
    edge_support = cv2.dilate(sig.edges01.astype(np.uint8), _k(3), iterations=1)
    bad = (
        (boundary > 0)
        & (thin > 0)
        & (reach_bg > 0)
        & (((bg_like > 0) | (shadow > 0)))
        & (edge_support == 0)
    )
    bad_ratio = float(np.sum(bad)) / float(boundary.sum() + 1e-6)
    if bad_ratio < float(cfg.global_halo_peel_trigger_ratio):
        return m

    allow = (m & (1 - bad.astype(np.uint8))).astype(np.uint8)
    recon = _geodesic_reconstruct(core, allow, max_iters=160)
    recon = _select_component_centered(recon)
    rm_frac = float(max(0.0, float(m.sum()) - float(recon.sum()))) / float(m.sum() + 1e-6)
    if rm_frac <= 0.0:
        return m
    if rm_frac > float(cfg.global_halo_peel_max_remove_frac):
        return m

    if runtime is not None:
        runtime["halo_peel"] = {
            "bad_ratio": float(bad_ratio),
            "removed_px": float(max(0.0, float(m.sum()) - float(recon.sum()))),
            "removed_frac": float(rm_frac),
        }
    return recon.astype(np.uint8)


def _tip_restore(
    mask01: np.ndarray,
    sig: _SegSignals,
    cfg: SegmentationConfig,
    runtime: dict[str, object] | None = None,
) -> np.ndarray:
    m = (mask01 > 0).astype(np.uint8)
    if m.sum() < 500 or not bool(cfg.tip_restore_enabled):
        return m

    h, w = m.shape[:2]
    border_vals = sig.d_bg_u8[sig.border_mask01 > 0]
    if border_vals.size < 80:
        return m
    fg_thr = float(np.percentile(border_vals, 99.0) + 8.0)
    fg_thr = float(max(15.0, min(240.0, fg_thr)))
    reach_bg = _reachable_background_mask(sig, strictness=float(cfg.shadow_strictness), cfg=cfg, roi01=None)

    ring = (cv2.dilate(m, _k(3), iterations=1) > 0) & (m == 0)
    edge = cv2.dilate(sig.edges01.astype(np.uint8), _k(3), iterations=1) > 0
    fg_like = sig.d_bg_u8.astype(np.float32) >= float(fg_thr)
    add = ring & edge & fg_like & (reach_bg == 0)
    add_px = int(np.sum(add))
    if add_px <= 0:
        return m

    max_add = float(cfg.tip_restore_max_add_frac) * float(m.sum() + 1e-6)
    if float(add_px) > max_add:
        return m

    out = m.copy()
    out[add] = 1
    out = _select_component_centered(out.astype(np.uint8))
    if runtime is not None:
        runtime["tip_restore"] = {
            "added_px": float(add_px),
            "added_frac": float(float(add_px) / float(m.sum() + 1e-6)),
        }
    return out.astype(np.uint8)


def _largest_component(mask01: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask01.astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask01
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = 1 + int(np.argmax(areas))
    return (labels == best_idx).astype(np.uint8)


def _edge_based_finger_refine(
    bgr: np.ndarray,
    mask01: np.ndarray,
    sig: "_SegSignals",
    cfg: SegmentationConfig,
    runtime: dict[str, object] | None = None,
) -> np.ndarray:
    """
    Edge-based finger separation and tip recovery.

    Phase 1 – Gap carving (inter-finger fix):
        Compute convex hull gap. Identify mask pixels that lie in the
        gap's boundary band, are thin (low distance transform), and are
        background-like or have low contrast with local background.
        Carve them out to separate fingers.

    Phase 2 – Tip recovery (leather/latex fix):
        Find strong edges just outside mask boundary → grow mask outward
        to include edge-supported, non-background pixels.
    """
    m = (mask01 > 0).astype(np.uint8)
    h, w = m.shape[:2]
    area0 = int(m.sum())
    if area0 < 500:
        return m

    # Tip protection: never let this stage carve away a fingertip band that was
    # already present. This is especially important for fabric, where gap carving
    # can cascade once tips are slightly clipped upstream.
    tip_keep = np.zeros_like(m, dtype=np.uint8)
    if bool(getattr(cfg, "tip_protect_enabled", True)):
        frac = float(getattr(cfg, "tip_protect_frac", 0.18))
        kd = float(getattr(cfg, "tip_protect_dilate_frac", 0.006))
        tip_keep = _tip_keep_mask_upright(bgr, m, frac=frac, dilate_frac=kd)
        tip_keep = (tip_keep & m).astype(np.uint8)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    profile = str(cfg.profile_name).lower()
    carved_px = 0
    grown_px = 0
    m_in = m.copy()
    top_in = None
    if profile == "fabric":
        ys = np.where(m_in > 0)[0]
        top_in = int(ys.min()) if ys.size else None

    # ──────────────────────────────────────────────────────────────────
    #  Phase 1: Inter-finger gap carve (fabric only).
    #  Strategy: pixels in [mask AND hull-gap-region AND thin-boundary
    #  AND (bg-like OR shadow-like OR edge-near)] get carved out.
    #  Only runs for fabric — leather/latex don't have this problem
    #  and the criteria over-carves their finger boundary shadows.
    # ──────────────────────────────────────────────────────────────────
    if profile == "fabric":
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            hull = cv2.convexHull(cnt)
            hull_mask = np.zeros((h, w), np.uint8)
            cv2.drawContours(hull_mask, [hull], -1, 1, -1)
            gap = ((hull_mask > 0) & (m == 0)).astype(np.uint8)
            gap_frac = float(gap.sum()) / float(area0 + 1e-6)

            if gap_frac > 0.02:
                # Dilate the gap to identify the "gap influence zone" — mask
                # pixels immediately adjacent to detectable inter-finger gaps.
                gap_dil_k = max(7, int(round(min(h, w) * 0.018)))
                gap_zone = cv2.dilate(gap, _k(gap_dil_k), iterations=1)
                gap_zone = ((gap_zone > 0) & (m > 0)).astype(np.uint8)

                # Distance transform: only carve thin (boundary-like) pixels.
                dt = cv2.distanceTransform((m * 255).astype(np.uint8), cv2.DIST_L2, 5)
                thin_thr = max(6.0, min(h, w) * 0.018)
                thin = (dt <= thin_thr).astype(np.uint8)

                # Background-like check (relaxed for shadow regions).
                border_vals = sig.d_bg_u8[sig.border_mask01 > 0]
                if border_vals.size >= 80:
                    bg_thr = float(np.percentile(border_vals, float(cfg.bg_like_percentile)))
                    bg_thr = float(bg_thr + float(cfg.bg_like_margin) * 1.5)
                else:
                    bg_thr = float(_bg_like_thr_u8(sig)) * 1.3
                bg_like = (sig.d_bg_u8.astype(np.float32) <= bg_thr).astype(np.uint8)

                # Shadow-like check: low saturation pixels in inter-finger
                # areas tend to be desaturated shadows.
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                low_sat = (hsv[:, :, 1] < 35).astype(np.uint8)

                # Carve: gap_zone AND thin AND (bg-like OR low-saturation).
                carve = (
                    (gap_zone > 0)
                    & (thin > 0)
                    & ((bg_like > 0) | (low_sat > 0))
                ).astype(np.uint8)

                # Edge-boost: also carve thin gap-zone pixels near edges
                # (finger outlines).
                edges = cv2.Canny(gray, 50, 150)
                edge_zone = cv2.dilate(edges, _k(3), iterations=1)
                carve_edge = (
                    (gap_zone > 0) & (thin > 0) & (edge_zone > 0)
                ).astype(np.uint8)
                carve = ((carve > 0) | (carve_edge > 0)).astype(np.uint8)
                if int(tip_keep.sum()) > 0:
                    carve = (carve & (1 - tip_keep)).astype(np.uint8)

                rm_frac = float(carve.sum()) / float(area0 + 1e-6)
                if 0.001 < rm_frac < 0.10:
                    m[carve > 0] = 0
                    m = _select_component_centered(m)
                    # Safety: if this carve trims fingertips (common failure mode),
                    # revert the carve rather than cascading into tip loss.
                    if top_in is not None:
                        ys2 = np.where(m > 0)[0]
                        top_after = int(ys2.min()) if ys2.size else None
                        if top_after is None:
                            m = m_in.copy()
                        else:
                            thr = max(8, int(round(0.020 * float(h))))
                            if int(top_after - top_in) > thr:
                                m = m_in.copy()
                    carved_px = int(carve.sum()) if int((m != m_in).any()) else 0

    # ──────────────────────────────────────────────────────────────────
    #  Phase 2: Tip recovery — grow mask outward where edges confirm
    #  glove boundary (rescues clipped fingertips).
    #  Skip for fabric: Phase 2 would grow INTO gaps, making it worse.
    # ──────────────────────────────────────────────────────────────────
    if profile != "fabric":
        grow_k = max(5, int(round(min(h, w) * 0.025)))
        outer_ring = cv2.dilate(m, _k(grow_k), iterations=1)
        outer_ring = ((outer_ring > 0) & (m == 0)).astype(np.uint8)

        edges_strong = cv2.Canny(gray, 60, 160)
        edge_support = cv2.dilate(edges_strong, _k(3), iterations=1)

        border_vals = sig.d_bg_u8[sig.border_mask01 > 0]
        if border_vals.size >= 80:
            bg_thr = float(np.percentile(border_vals, float(cfg.bg_like_percentile)))
            bg_thr = float(bg_thr + float(cfg.bg_like_margin))
        else:
            bg_thr = float(_bg_like_thr_u8(sig))
        not_bg = (sig.d_bg_u8.astype(np.float32) > bg_thr).astype(np.uint8)

        grow_pixels = (
            (outer_ring > 0)
            & (edge_support > 0)
            & (not_bg > 0)
        ).astype(np.uint8)

        grow_area = int(grow_pixels.sum())
        max_grow = max(200, int(area0 * 0.04))
        if 0 < grow_area <= max_grow:
            m = ((m > 0) | (grow_pixels > 0)).astype(np.uint8)
            m = _select_component_centered(m)
            grown_px = grow_area

    if runtime is not None:
        runtime["edge_finger_refine"] = {
            "gap_carved_px": carved_px,
            "tip_grown_px": grown_px,
        }

    if int(tip_keep.sum()) > 0:
        m = ((m > 0) | (tip_keep > 0)).astype(np.uint8)
        m = _select_component_centered(m)
    return m


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


def _prune_boundary_color_outliers(
    bgr: np.ndarray,
    mask01: np.ndarray,
    *,
    cfg: SegmentationConfig | None = None,
    sig: _SegSignals | None = None,
    stats: dict[str, float | bool | str] | None = None,
) -> np.ndarray:
    """
    Remove likely background leakage on the boundary by checking LAB color
    consistency against an eroded interior region.
    """
    m = (mask01 > 0).astype(np.uint8)
    if stats is not None:
        stats.clear()
        stats.update(
            {
                "applied": False,
                "reverted": False,
                "reason": "none",
                "area_before": float(m.sum()),
                "area_after": float(m.sum()),
                "border_touch_before": float(_border_touch_ratio(m)),
                "border_touch_after": float(_border_touch_ratio(m)),
            }
        )
    if m.sum() < 100:
        if stats is not None:
            stats["reason"] = "small_mask"
        return m
    if cfg is not None and not bool(cfg.boundary_prune_enabled):
        if stats is not None:
            stats["reason"] = "disabled"
        return m

    bt0 = float(_border_touch_ratio(m))
    if (
        cfg is not None
        and bool(cfg.boundary_prune_skip_if_low_leak)
        and str(cfg.profile_name).lower() == "latex"
        and bt0 <= 0.03
    ):
        if stats is not None:
            stats["reason"] = "low_leak_latex_skip"
        return m

    h, w = m.shape[:2]
    erode_sz = max(3, int(round(min(h, w) * 0.012)))
    core = cv2.erode(m, _k(erode_sz), iterations=1)
    if core.sum() < 80:
        if stats is not None:
            stats["reason"] = "small_core"
        return m

    boundary = (m - core).astype(np.uint8)
    if boundary.sum() < 20:
        if stats is not None:
            stats["reason"] = "small_boundary"
        return m

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    med = np.median(lab[core > 0], axis=0)
    de = np.linalg.norm(lab - med.reshape(1, 1, 3), axis=2)
    core_de = de[core > 0]
    if core_de.size == 0:
        if stats is not None:
            stats["reason"] = "core_stats_empty"
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
    if cfg is None or bool(cfg.boundary_prune_edge_rescue):
        if sig is not None:
            edge01 = (sig.edges01 > 0).astype(np.uint8)
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            edge01 = (_auto_canny(gray, sigma=0.33) > 0).astype(np.uint8)
        edge_band = cv2.dilate(edge01, _k(max(3, erode_sz | 1)), iterations=1)
        keep_boundary = (keep_boundary | (((boundary > 0) & (edge_band > 0)).astype(np.uint8))).astype(np.uint8)
    core_l = lab[:, :, 0][core > 0]
    if core_l.size and float(np.mean(core_l)) > 115.0:
        l_lo = float(np.percentile(core_l, 5) - 18.0)
        keep_boundary = (keep_boundary & (lab[:, :, 0] >= l_lo)).astype(np.uint8)

    out = (core | keep_boundary).astype(np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, _k(max(5, erode_sz + 2)), iterations=1)
    out = _largest_component(out)
    bt1 = float(_border_touch_ratio(out))
    area0 = float(m.sum())
    area1 = float(out.sum())
    removed_frac = float(max(0.0, area0 - area1)) / float(area0 + 1e-6)
    leak_improve = float(bt0 - bt1)
    # If pruning removed too much area without meaningful leakage improvement, revert.
    if removed_frac > 0.06 and leak_improve < 0.010:
        if stats is not None:
            stats["applied"] = True
            stats["reverted"] = True
            stats["reason"] = "revert_large_drop_low_gain"
            stats["area_after"] = float(area0)
            stats["border_touch_after"] = float(bt0)
        return m
    if stats is not None:
        stats["applied"] = True
        stats["reason"] = "ok"
        stats["area_after"] = float(area1)
        stats["border_touch_after"] = float(bt1)
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


def _mask_area_frac(mask01: np.ndarray) -> float:
    h, w = mask01.shape[:2]
    return float(np.sum(mask01 > 0)) / float((h * w) + 1e-6)


def _shape_plausibility(mask01: np.ndarray) -> tuple[float, dict[str, float | int]]:
    """
    Lightweight glove-likeness score used only as a *tie-breaker*.

    The goal is to prefer glove-like silhouettes over generic blobs when multiple
    candidates have similar evidence/edge/color scores.
    """
    m = (mask01 > 0).astype(np.uint8)
    h, w = m.shape[:2]
    if int(m.sum()) < 120:
        return 0.0, {"shape_solidity": 0.0, "shape_defects": 0, "shape_score": 0.0}

    cnts, _hier = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, {"shape_solidity": 0.0, "shape_defects": 0, "shape_score": 0.0}
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area <= 1.0:
        return 0.0, {"shape_solidity": 0.0, "shape_defects": 0, "shape_score": 0.0}

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area / float(hull_area + 1e-6))

    # Convexity defects (finger valleys) are a weak glove cue, but folded gloves
    # can have zero defects, so keep this influence small.
    defects_count = 0
    try:
        hull_idx = cv2.convexHull(cnt, returnPoints=False)
        if hull_idx is not None and len(hull_idx) >= 4 and len(cnt) >= 4:
            defects = cv2.convexityDefects(cnt, hull_idx)
            if defects is not None and len(defects) > 0:
                depth_px = defects[:, 0, 3].astype(np.float32) / 256.0
                thr = float(max(2.0, 0.020 * float(min(h, w))))
                defects_count = int(np.sum(depth_px >= thr))
    except Exception:
        defects_count = 0

    # Prefer moderately non-convex shapes (typical gloves) over near-rectangular blobs.
    sol_score = float(np.clip(1.0 - abs(solidity - 0.88) / 0.20, 0.0, 1.0))
    def_score = float(np.clip(float(defects_count) / 4.0, 0.0, 1.0))
    shape_score = float((0.65 * sol_score) + (0.35 * def_score))
    return shape_score, {"shape_solidity": float(solidity), "shape_defects": int(defects_count), "shape_score": float(shape_score)}


def _mask_iou(a01: np.ndarray, b01: np.ndarray) -> float:
    a = (a01 > 0).astype(np.uint8)
    b = (b01 > 0).astype(np.uint8)
    inter = float(np.sum((a > 0) & (b > 0)))
    union = float(np.sum((a > 0) | (b > 0)))
    return float(inter / (union + 1e-6))


def _tip_band_mask(mask01: np.ndarray, *, frac: float) -> np.ndarray:
    """
    Return a 0/1 mask selecting the likely fingertip-end band of the silhouette.

    Uses PCA along the major axis, then chooses the narrower end (cuff is typically
    wider than finger tips). This is used as a *soft keep* region so refinement
    passes don't trim tips that were already present in a candidate.
    """
    m = (mask01 > 0).astype(np.uint8)
    if int(m.sum()) < 200:
        return np.zeros_like(m, dtype=np.uint8)

    ys, xs = np.where(m > 0)
    if xs.size < 200:
        return np.zeros_like(m, dtype=np.uint8)

    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = pts.mean(axis=0)
    pts0 = pts - mean
    cov = (pts0.T @ pts0) / float(max(1, pts0.shape[0]))
    try:
        vals, vecs = np.linalg.eigh(cov)
    except Exception:
        return np.zeros_like(m, dtype=np.uint8)

    v = vecs[:, int(np.argmax(vals))].astype(np.float32)
    nv = float(np.linalg.norm(v)) + 1e-6
    v = v / nv
    u = np.array([-v[1], v[0]], dtype=np.float32)

    proj = (pts0 @ v).astype(np.float32)
    ortho = (pts0 @ u).astype(np.float32)
    pmin = float(np.min(proj))
    pmax = float(np.max(proj))
    rng = float(pmax - pmin)
    if not np.isfinite(rng) or rng < 1e-3:
        return np.zeros_like(m, dtype=np.uint8)

    f = float(max(0.06, min(0.35, frac)))
    end_band = float(max(0.10, min(0.22, 0.12 if f <= 0.18 else (0.10 + 0.6 * f))))
    sel_low = proj <= float(pmin + end_band * rng)
    sel_high = proj >= float(pmax - end_band * rng)
    if int(np.sum(sel_low)) < 30 or int(np.sum(sel_high)) < 30:
        # Fallback: pick an arbitrary end.
        tip_is_high = True
    else:
        # Narrower end is more likely to be fingertips than the cuff.
        w_low = float(np.percentile(np.abs(ortho[sel_low]), 95) - np.percentile(np.abs(ortho[sel_low]), 5))
        w_high = float(np.percentile(np.abs(ortho[sel_high]), 95) - np.percentile(np.abs(ortho[sel_high]), 5))
        tip_is_high = bool(w_high <= w_low)

    if tip_is_high:
        keep = proj >= float(pmax - f * rng)
    else:
        keep = proj <= float(pmin + f * rng)

    tip = np.zeros_like(m, dtype=np.uint8)
    tip[ys[keep], xs[keep]] = 1
    return tip.astype(np.uint8)


def _tip_keep_mask_upright(
    bgr: np.ndarray,
    mask01: np.ndarray,
    *,
    frac: float,
    dilate_frac: float,
) -> np.ndarray:
    """
    Tip keep-mask derived in an upright-normalized frame (fingers on top).

    This is more robust than guessing which PCA-end is the fingertips in the
    original frame. Output is in the original image coordinates (0/1 uint8).
    """
    m0 = (mask01 > 0).astype(np.uint8)
    h0, w0 = m0.shape[:2]
    if int(m0.sum()) < 200:
        return np.zeros_like(m0, dtype=np.uint8)

    rot_bgr, rot_m01, mat = _upright_normalize(bgr, m0)
    x, y, ww, hh = cv2.boundingRect(rot_m01.astype(np.uint8))
    if ww <= 0 or hh <= 0:
        return np.zeros_like(m0, dtype=np.uint8)

    f = float(max(0.06, min(0.35, frac)))
    band_h = int(max(6, round(f * float(hh))))
    y_end_top = int(min(rot_m01.shape[0], max(0, y + band_h)))
    y_start_bot = int(max(0, min(rot_m01.shape[0], (y + hh) - band_h)))
    tip_rot = np.zeros_like(rot_m01, dtype=np.uint8)
    # Protect both ends. We intentionally don't rely on the fingers-on-top
    # heuristic here; some glove shapes make that ambiguous.
    tip_rot[y:y_end_top, x : x + ww] = 1
    tip_rot[y_start_bot : y + hh, x : x + ww] = 1
    tip_rot = (tip_rot & rot_m01).astype(np.uint8)

    if int(tip_rot.sum()) > 0:
        kd = max(3, int(round(float(dilate_frac) * float(min(rot_m01.shape[:2])))))
        kd = int(min(19, max(3, kd | 1)))
        tip_rot = cv2.dilate(tip_rot, _k(kd), iterations=1).astype(np.uint8)

    inv = cv2.invertAffineTransform(mat)
    tip_back = cv2.warpAffine(tip_rot, inv, (w0, h0), flags=cv2.INTER_NEAREST, borderValue=0)
    tip_back = (tip_back > 0).astype(np.uint8)
    if int(tip_back.sum()) > 0:
        kd2 = max(3, int(round(float(dilate_frac) * float(min(h0, w0)))))
        kd2 = int(min(19, max(3, kd2 | 1)))
        tip_back = cv2.dilate(tip_back, _k(kd2), iterations=1).astype(np.uint8)
    return tip_back.astype(np.uint8)


def _build_silhouette_ref(cand_lo: np.ndarray, cand_edge: np.ndarray, cand_edge_s: np.ndarray, h: int, w: int) -> np.ndarray:
    sil = ((cand_lo > 0) | (cand_edge > 0) | (cand_edge_s > 0)).astype(np.uint8)
    if sil.sum() < int(0.02 * h * w):
        return sil
    k = max(5, int(round(min(h, w) * 0.012)))
    sil = cv2.morphologyEx(sil, cv2.MORPH_OPEN, _k(max(3, k // 2)), iterations=1)
    sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, _k(k), iterations=2)
    sil = _select_component_centered(_largest_component(sil))
    return sil.astype(np.uint8)


def _auto_canny(img_u8: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    v = float(np.median(img_u8))
    if v < 1.0:
        return cv2.Canny(img_u8, 30, 90)
    lo = int(max(0, (1.0 - float(sigma)) * v))
    hi = int(min(255, (1.0 + float(sigma)) * v))
    hi = max(hi, lo + 10)
    return cv2.Canny(img_u8, lo, hi)


def _shadow_like_mask_internal(sig: _SegSignals, *, strictness: float, suppress_center: bool) -> np.ndarray:
    """
    Detect "shadow-like background" pixels: chroma close to border background but
    noticeably darker in L.

    This is used only as a *guard* (seed sure-background) around the glove to
    prevent mask expansion into shadows.
    """
    lab = sig.lab_u8.astype(np.float32)
    l = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    bg_l, bg_a, bg_b = float(sig.bg_med[0]), float(sig.bg_med[1]), float(sig.bg_med[2])
    mad_a, mad_b = float(sig.bg_mad[1]), float(sig.bg_mad[2])
    mad_a = max(2.0, mad_a)
    mad_b = max(2.0, mad_b)

    ab = np.sqrt(((a - bg_a) / mad_a) ** 2 + ((b - bg_b) / mad_b) ** 2).astype(np.float32)
    ldiff = (bg_l - l).astype(np.float32)  # positive => darker than border background

    border = sig.border_mask01 > 0
    ab_b = ab[border]
    ldiff_b = ldiff[border]
    if ab_b.size < 100 or ldiff_b.size < 100:
        return np.zeros_like(sig.border_mask01, dtype=np.uint8)

    strict = float(max(0.5, min(3.0, strictness)))
    thr_ab = float(np.percentile(ab_b, 88) + (0.6 / strict))
    thr_ab = float(max(0.8, min(4.2, thr_ab)))
    thr_l = float(np.percentile(ldiff_b, 92) + (4.0 * strict))
    thr_l = float(max(8.0, min(38.0, thr_l)))

    shadow = (ab <= thr_ab) & (ldiff >= thr_l)
    # Avoid marking the central area too aggressively; shadows are typically around the glove.
    # For webbing/gap carving we use a different rule (see callers).
    if suppress_center:
        shadow &= (sig.center_prior < 0.92)
    return shadow.astype(np.uint8)


def _reachable_background_mask(
    sig: _SegSignals,
    *,
    strictness: float,
    cfg: SegmentationConfig | None = None,
    roi01: np.ndarray | None = None,
) -> np.ndarray:
    """
    Estimate border-reachable background + shadow region using a chroma-first model.

    Rationale: on white backgrounds, shadows change L strongly but often keep a,b close
    to background. Fabric/latex gloves can be low-chroma too, so we also require L to
    stay within a margin of the border background (allowing shadow but excluding dark gloves).

    Returns a 0/1 mask of pixels connected to the border.
    """
    lab = sig.lab_u8.astype(np.float32)
    l = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    bg_l, bg_a, bg_b = float(sig.bg_med[0]), float(sig.bg_med[1]), float(sig.bg_med[2])
    mad_a, mad_b = float(sig.bg_mad[1]), float(sig.bg_mad[2])
    mad_a = max(2.0, mad_a)
    mad_b = max(2.0, mad_b)

    ab = np.sqrt(((a - bg_a) / mad_a) ** 2 + ((b - bg_b) / mad_b) ** 2).astype(np.float32)
    border = sig.border_mask01 > 0
    ab_b = ab[border]
    if ab_b.size < 120:
        return sig.border_mask01.astype(np.uint8)

    strict = float(max(0.5, min(3.0, strictness)))
    thr_ab = float(np.percentile(ab_b, 96.0) + (0.9 / strict))
    thr_ab = float(max(0.8, min(5.0, thr_ab)))

    # Allow some L drop for shadow, but keep it bounded to avoid including dark glove pixels.
    # When strictness is lower, allow more shadow.
    l_margin = float(42.0 + (20.0 / strict))
    prof = str(cfg.profile_name).lower() if cfg is not None else "balanced"
    if prof == "leather":
        l_margin -= 7.0
    l_margin = float(max(32.0, min(90.0, l_margin)))
    l_ok = l >= float(bg_l - l_margin)
    # Latex/fabric deep shadows between fingers can be dark but still border-reachable background.
    # Widen L-margin only in the finger ROI to avoid global over-carving.
    if prof in {"latex", "fabric"}:
        roi = np.ones_like(l_ok, dtype=np.uint8) if roi01 is None else (roi01 > 0).astype(np.uint8)
        l_margin_roi = float(min(104.0, l_margin + 12.0))
        l_ok_roi = l >= float(bg_l - l_margin_roi)
        l_ok = np.where(roi > 0, l_ok_roi, l_ok)

    cand = ((ab <= thr_ab) & (l_ok > 0)).astype(np.uint8)

    # Keep only border-connected part.
    reachable = _border_connected(cand, sig.border_mask01.astype(np.uint8))
    return reachable.astype(np.uint8)


def _shadow_like_mask(sig: _SegSignals, *, strictness: float = 1.0) -> np.ndarray:
    return _shadow_like_mask_internal(sig, strictness=float(strictness), suppress_center=True)


def _shadow_like_mask_anywhere(sig: _SegSignals, *, strictness: float = 1.0) -> np.ndarray:
    """
    Shadow-like mask without center prior suppression. Useful for penalizing
    masks that include shadow/background pockets (e.g., thumb-index web space).
    """
    return _shadow_like_mask_internal(sig, strictness=float(strictness), suppress_center=False)


def _geodesic_reconstruct(seed01: np.ndarray, allow01: np.ndarray, max_iters: int) -> np.ndarray:
    """
    Geodesic dilation: iteratively dilate seed inside allow until convergence.
    Both inputs are 0/1 uint8.
    """
    seed = (seed01 > 0).astype(np.uint8)
    allow = (allow01 > 0).astype(np.uint8)
    k3 = _k(3)
    cur = (seed & allow).astype(np.uint8)
    for _ in range(int(max_iters)):
        nxt = (cv2.dilate(cur, k3, iterations=1) & allow).astype(np.uint8)
        if int(nxt.sum()) == int(cur.sum()):
            break
        cur = nxt
    return cur.astype(np.uint8)


def _convexity_webbing_carve(
    upright_mask01: np.ndarray,
    roi01: np.ndarray,
    bbox: tuple[int, int, int, int],
    reach_bg01: np.ndarray,
    cfg: SegmentationConfig,
    *,
    stats: dict[str, float | str] | None = None,
) -> np.ndarray:
    """
    Shape-only carve aimed at removing the thumb-index webbing fill.

    If the silhouette is overly convex (high solidity), compute convexity defects and
    carve 1-2 deepest defect wedges in the finger ROI.

    Updated:
    - evidence mode: require overlap with border-reachable background.
    - geometry mode: depth/ROI-only carve with strict safety caps.
    - hybrid mode: evidence-first, then geometry fallback if evidence is insufficient.
    """
    if stats is not None:
        stats.clear()
        stats.update(
            {
                "mode_requested": str(cfg.gap_carve_mode),
                "mode_used": "none",
                "solidity": 0.0,
                "evidence_removed_px": 0.0,
                "geometry_removed_px": 0.0,
            }
        )
    m = (upright_mask01 > 0).astype(np.uint8)
    if m.sum() < 600:
        return m
    x, y, ww, hh = bbox
    if ww < 60 or hh < 80:
        return m

    # Solidity check.
    cnts, _ = cv2.findContours((m * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return m
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    hull_pts = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull_pts)) + 1e-6
    solidity = float(area / hull_area)
    strict_solidity = bool(solidity >= float(cfg.webbing_solidity_thr))
    if stats is not None:
        stats["solidity"] = float(solidity)

    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 4:
        return m
    try:
        defects = cv2.convexityDefects(cnt, hull_idx)
    except cv2.error:
        # Rarely, a noisy contour can trigger non-monotonous hull indices.
        # Smooth/approximate once and retry; if still invalid, skip this carve pass.
        cnt2 = cv2.approxPolyDP(cnt, epsilon=2.0, closed=True)
        if cnt2 is None or len(cnt2) < 4:
            return m
        hull_idx2 = cv2.convexHull(cnt2, returnPoints=False)
        if hull_idx2 is None or len(hull_idx2) < 4:
            return m
        try:
            defects = cv2.convexityDefects(cnt2, hull_idx2)
            cnt = cnt2
            hull_idx = hull_idx2
        except cv2.error:
            return m
    if defects is None:
        return m

    min_depth = float(max(8.0, float(cfg.webbing_depth_frac) * float(min(ww, hh)) * 256.0))
    # Exclude cuff/collar concavities: webbing gap is usually below the top ~22% of bbox.
    y_min = int(round(float(y) + 0.22 * float(hh)))
    # Defects: [start_idx, end_idx, far_idx, depth*256]
    candidates = []
    for d in defects[:, 0, :]:
        si, ei, fi, depth = int(d[0]), int(d[1]), int(d[2]), float(d[3])
        if depth < min_depth:
            continue
        far = cnt[fi][0]
        fx, fy = int(far[0]), int(far[1])
        if roi01[fy, fx] == 0:
            continue
        # Focus on upper portion where fingers/thumb are.
        if fy > int(round(y + 0.70 * float(hh))):
            continue
        if fy < y_min:
            continue
        candidates.append((depth, si, ei, fi, fx, fy))

    if not candidates:
        return m
    # Determine likely thumb side in upright frame (left vs right) using a mid-height band.
    cx = float(x) + 0.5 * float(ww)
    y0 = int(round(float(y) + 0.35 * float(hh)))
    y1 = int(round(float(y) + 0.65 * float(hh)))
    y0 = max(int(y), min(int(y + hh - 1), y0))
    y1 = max(y0 + 1, min(int(y + hh), y1))
    left_d: list[float] = []
    right_d: list[float] = []
    for yy in range(y0, y1):
        row = m[yy, x : x + ww]
        xs = np.where(row > 0)[0]
        if xs.size < 5:
            continue
        xl = float(x + int(xs[0]))
        xr = float(x + int(xs[-1]))
        left_d.append(float(cx - xl))
        right_d.append(float(xr - cx))
    thumb_is_right = bool(np.median(right_d) > np.median(left_d)) if left_d and right_d else True

    # Build wedges and rank by reach_bg overlap (and depth), focusing on thumb side.
    reach_bg = (reach_bg01 > 0).astype(np.uint8)
    mode = str(cfg.gap_carve_mode).strip().lower()
    if mode not in {"evidence", "geometry", "hybrid"}:
        mode = "hybrid"

    def _wedge_overlap(si: int, ei: int, fi: int) -> tuple[np.ndarray, float, float]:
        s = cnt[si][0]
        e = cnt[ei][0]
        f = cnt[fi][0]
        poly = np.array([[s[0], s[1]], [f[0], f[1]], [e[0], e[1]]], dtype=np.int32)
        wedge = np.zeros_like(m, dtype=np.uint8)
        cv2.fillConvexPoly(wedge, poly, 1)
        if int(cfg.webbing_wedge_dilate_px) > 0:
            wedge = cv2.dilate(wedge, _k(int(cfg.webbing_wedge_dilate_px)), iterations=1)
        wedge = (wedge > 0).astype(np.uint8)
        w_roi = (wedge > 0) & (roi01 > 0)
        denom = float(np.sum(w_roi))
        if denom < 40:
            return wedge, 0.0, denom
        ov = float(np.sum(w_roi & (reach_bg > 0))) / float(denom + 1e-6)
        return wedge, ov, denom

    cand_scored = []
    for depth, si, ei, fi, fx, fy in candidates:
        side_ok = (fx > cx) if thumb_is_right else (fx < cx)
        if not side_ok:
            continue
        wedge, ov, _denom = _wedge_overlap(si, ei, fi)
        y_norm = (float(fy) - float(y)) / float(hh + 1e-6)
        loc_bonus = float(max(0.55, 1.0 - abs(y_norm - 0.48)))
        score_e = float(ov) * float(depth) * float(loc_bonus)
        score_g = float(depth) * float(loc_bonus)
        cand_scored.append((score_e, score_g, float(ov), depth, si, ei, fi, fx, fy, wedge))

    # Fallback: if thumb-side filter produced nothing, consider all candidates.
    if not cand_scored:
        for depth, si, ei, fi, fx, fy in candidates:
            wedge, ov, _denom = _wedge_overlap(si, ei, fi)
            y_norm = (float(fy) - float(y)) / float(hh + 1e-6)
            loc_bonus = float(max(0.55, 1.0 - abs(y_norm - 0.48)))
            score_e = float(ov) * float(depth) * float(loc_bonus)
            score_g = float(depth) * float(loc_bonus)
            cand_scored.append((score_e, score_g, float(ov), depth, si, ei, fi, fx, fy, wedge))

    if not cand_scored:
        return m

    cand_scored_e = sorted(cand_scored, reverse=True, key=lambda t: float(t[0]))
    cand_scored_g = sorted(cand_scored, reverse=True, key=lambda t: float(t[1]))
    # If not very convex, only allow a single carve unless evidence is strong.
    max_def = int(max(1, cfg.webbing_max_defects)) if strict_solidity else 1
    cand_scored_e = cand_scored_e[:max_def]
    cand_scored_g = cand_scored_g[:max_def]

    dt = cv2.distanceTransform((m * 255).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    # Allow carving slightly deeper than a pure boundary band: webbing gaps can be fairly wide.
    strength = float(max(0.5, min(2.5, float(cfg.reach_carve_strength))))
    dt_thr = float(max(14.0, (0.060 + 0.020 * strength) * float(min(ww, hh))))
    dt_thr = float(min(dt_thr, 0.24 * float(min(ww, hh))))

    out = m.copy()
    evidence_removed = 0
    if mode in {"evidence", "hybrid"}:
        if reach_bg.sum() > 100:
            for _score_e, _score_g, overlap, _depth, _si, _ei, _fi, _fx, _fy, wedge in cand_scored_e:
                thr_overlap = float(cfg.webbing_reach_overlap_thr)
                if not strict_solidity:
                    thr_overlap = float(max(thr_overlap, 0.26))
                if overlap < thr_overlap:
                    continue
                rm = (wedge > 0) & (roi01 > 0) & (reach_bg > 0) & (dt <= dt_thr) & (out > 0)
                rm_px = int(np.sum(rm))
                if rm_px < 25:
                    continue
                out[rm] = 0
                evidence_removed += rm_px
        if evidence_removed > 0:
            out = _select_component_centered(out.astype(np.uint8))
    if stats is not None:
        stats["evidence_removed_px"] = float(evidence_removed)

    geometry_removed = 0
    need_geometry = False
    if mode == "geometry":
        need_geometry = True
    elif mode == "hybrid":
        rm_frac = float(evidence_removed) / float(m.sum() + 1e-6)
        need_geometry = bool((rm_frac < 0.002) and (solidity >= float(cfg.gap_geometry_fallback_solidity)))

    if need_geometry:
        pre = out.copy() if mode == "hybrid" else m.copy()
        base = pre.copy()
        bt0 = float(_border_touch_ratio(base))
        x0, y0, ww0, hh0 = cv2.boundingRect(base.astype(np.uint8))
        ext0 = float(base.sum()) / float(ww0 * hh0 + 1e-6) if ww0 > 0 and hh0 > 0 else 0.0
        cap = float(cfg.gap_max_remove_frac) * float(base.sum() + 1e-6)
        for _score_e, _score_g, _overlap, _depth, _si, _ei, _fi, _fx, _fy, wedge in cand_scored_g:
            if geometry_removed >= cap:
                break
            rm = (wedge > 0) & (roi01 > 0) & (dt <= dt_thr) & (pre > 0)
            rm_px = int(np.sum(rm))
            if rm_px < 25:
                continue
            if float(geometry_removed + rm_px) > cap:
                continue
            pre[rm] = 0
            geometry_removed += rm_px
        if geometry_removed > 0:
            cand = _select_component_centered(pre.astype(np.uint8))
            bt1 = float(_border_touch_ratio(cand))
            x1, y1, ww1, hh1 = cv2.boundingRect(cand.astype(np.uint8))
            ext1 = float(cand.sum()) / float(ww1 * hh1 + 1e-6) if ww1 > 0 and hh1 > 0 else 0.0
            # Reject if carve degrades outer silhouette behavior.
            if (bt1 <= bt0 + 0.01) and (ext1 >= ext0 - 0.04):
                out = cand
            else:
                geometry_removed = 0

    if stats is not None:
        stats["geometry_removed_px"] = float(geometry_removed)
        if geometry_removed > 0:
            stats["mode_used"] = "geometry" if mode == "geometry" else "hybrid"
        elif evidence_removed > 0:
            stats["mode_used"] = "evidence" if mode == "evidence" else "hybrid"
        else:
            stats["mode_used"] = "none"
    return _select_component_centered(out.astype(np.uint8)).astype(np.uint8)


def _compute_signals(bgr: np.ndarray) -> _SegSignals:
    h, w = bgr.shape[:2]
    lab_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_f = lab_u8.astype(np.float32)

    border_w = max(12, int(round(min(h, w) * 0.09)))
    border = np.zeros((h, w), dtype=np.uint8)
    border[:border_w, :] = 1
    border[-border_w:, :] = 1
    border[:, :border_w] = 1
    border[:, -border_w:] = 1

    border_px = lab_f[border > 0]
    if border_px.shape[0] < 120:
        bg_med = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        bg_mad = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        bg_is_uniform = False
        bg_is_white = False
    else:
        bg_med = np.median(border_px, axis=0).astype(np.float32)
        bg_mad = np.median(np.abs(border_px - bg_med.reshape(1, 3)), axis=0).astype(np.float32)
        bg_mad = np.maximum(bg_mad, np.array([3.0, 2.0, 2.0], dtype=np.float32))
        bg_std = border_px.std(axis=0)
        bg_is_uniform = bool(bg_std[0] < 16.0 and bg_std[1] < 7.0 and bg_std[2] < 7.0)
        bg_is_white = bool(bg_med[0] > 175.0 and bg_is_uniform)

    z = (lab_f - bg_med.reshape(1, 1, 3)) / bg_mad.reshape(1, 1, 3)
    d_bg = np.sqrt(np.sum(z * z, axis=2)).astype(np.float32)
    d_bg = cv2.GaussianBlur(d_bg, (5, 5), 0)
    d_bg_u8 = cv2.normalize(d_bg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    center_prior = _center_prior(h, w)

    # Gradient magnitude on CLAHE-normalized L channel.
    l = lab_u8[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    gx = cv2.Sobel(l2, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l2, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.GaussianBlur(mag, (5, 5), 0)
    grad_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Also compute edges on a,b channels: shadows typically change L but not chroma,
    # so ab-edges reduce shadow-boundary contamination.
    a = cv2.GaussianBlur(lab_u8[:, :, 1], (5, 5), 0)
    b = cv2.GaussianBlur(lab_u8[:, :, 2], (5, 5), 0)
    ax = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    ay = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    bx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
    by = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
    mag_ab = cv2.magnitude(ax, ay) + cv2.magnitude(bx, by)
    mag_ab = cv2.GaussianBlur(mag_ab, (5, 5), 0)
    grad_ab_u8 = cv2.normalize(mag_ab, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    edges_l = _auto_canny(grad_u8, sigma=0.33)
    edges_ab = _auto_canny(grad_ab_u8, sigma=0.33)
    edges_u8 = cv2.max(edges_l, edges_ab)
    grad_u8 = cv2.max(grad_u8, grad_ab_u8)
    edges01 = (edges_u8 > 0).astype(np.uint8)

    return _SegSignals(
        lab_u8=lab_u8,
        bg_med=bg_med,
        bg_mad=bg_mad,
        border_mask01=border.astype(np.uint8),
        center_prior=center_prior.astype(np.float32),
        d_bg=d_bg,
        d_bg_u8=d_bg_u8,
        grad_u8=grad_u8,
        edges01=edges01.astype(np.uint8),
        edges_u8=edges_u8.astype(np.uint8),
        bg_is_uniform=bool(bg_is_uniform),
        bg_is_white=bool(bg_is_white),
    )


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


def _threshold_segment_lab_l_bidirectional(l_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    l_blur = cv2.GaussianBlur(l_u8, (5, 5), 0)
    _, th = cv2.threshold(l_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_hi = (th > 0).astype(np.uint8)
    mask_lo = (th == 0).astype(np.uint8)
    return mask_hi, mask_lo


def _chroma_candidate(sig: _SegSignals) -> np.ndarray:
    """
    Candidate foreground from LAB chroma, useful when glove luminance is close to white background.
    """
    h, w = sig.lab_u8.shape[:2]
    lab_f = sig.lab_u8.astype(np.float32)
    ab = lab_f[:, :, 1:3] - 128.0
    chroma = np.sqrt((ab[:, :, 0] * ab[:, :, 0]) + (ab[:, :, 1] * ab[:, :, 1]))

    border_vals = chroma[sig.border_mask01 > 0]
    if border_vals.size < 80:
        return np.zeros((h, w), np.uint8)

    thr = float(np.percentile(border_vals, 99.2) + 2.0)
    thr = max(thr, float(np.percentile(chroma, 70.0)))
    fg = ((chroma >= thr) & (sig.center_prior > 0.04)).astype(np.uint8)

    k1 = max(3, int(round(min(h, w) * 0.006)))
    k2 = max(7, int(round(min(h, w) * 0.015)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _k(k1), iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _k(k2), iterations=2)
    fg = _select_component_centered(fg)
    return fg.astype(np.uint8)


def _texture_candidate(bgr: np.ndarray, sig: _SegSignals) -> np.ndarray:
    """
    Candidate foreground from texture energy. Helps on low-chroma fabric against white background.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    tex_u8 = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    border_vals = tex_u8[sig.border_mask01 > 0]
    if border_vals.size < 80:
        return np.zeros((h, w), np.uint8)

    thr = int(np.percentile(border_vals, 99.2) + 5.0)
    thr = max(thr, int(np.percentile(tex_u8, 78.0)))
    thr = int(min(250, max(16, thr)))
    fg = ((tex_u8 >= thr) & (sig.center_prior > 0.04)).astype(np.uint8)

    k1 = max(3, int(round(min(h, w) * 0.005)))
    k2 = max(5, int(round(min(h, w) * 0.011)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _k(k1), iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _k(k2), iterations=1)
    fg = _select_component_centered(fg)
    return fg.astype(np.uint8)


def _edge_alignment(mask01: np.ndarray, edges01: np.ndarray) -> float:
    if mask01.sum() < 50:
        return 0.0
    k = 5
    dil = cv2.dilate(mask01.astype(np.uint8), _k(k), iterations=1)
    ero = cv2.erode(mask01.astype(np.uint8), _k(k), iterations=1)
    band = (dil > 0).astype(np.uint8) - (ero > 0).astype(np.uint8)
    denom = float(band.sum()) + 1e-6
    return float(((band > 0) & (edges01 > 0)).sum()) / denom


def _bg_sep_u8(mask01: np.ndarray, d_bg_u8: np.ndarray, border01: np.ndarray) -> float:
    if mask01.sum() < 50:
        return 0.0
    in_vals = d_bg_u8[mask01 > 0]
    bd_vals = d_bg_u8[border01 > 0]
    if in_vals.size < 20 or bd_vals.size < 20:
        return 0.0
    return float(np.median(in_vals) - np.median(bd_vals))


def _score_mask(bgr: np.ndarray, mask01: np.ndarray, sig: _SegSignals, cfg: SegmentationConfig | None = None) -> tuple[float, dict]:
    h, w = mask01.shape[:2]
    area = float(mask01.sum())
    if area <= 0:
        return -1e9, {"reason": "empty"}

    # Reasonable area constraints.
    frac = area / float(h * w)
    if frac < 0.03 or frac > 0.90:
        return -1e9, {"reason": "area_out_of_range", "area_frac": float(frac)}

    x, y, ww, hh = cv2.boundingRect(mask01.astype(np.uint8))
    extent = area / float(ww * hh + 1e-6)

    # Penalize masks that hug the border too much (often background leakage).
    border_touch = _border_touch_ratio(mask01)
    if border_touch > 0.30:
        return -1e9, {"reason": "border_touch_high", "border_touch": float(border_touch)}

    # Background separation in border-distance space (stable for white backgrounds).
    bg_sep = _bg_sep_u8(mask01, sig.d_bg_u8, sig.border_mask01)

    # Centrality overlap (soft prior, not a hard gate).
    center_w = float((sig.center_prior * mask01.astype(np.float32)).sum()) / float(area + 1e-6)

    # Edge alignment: glove boundary should coincide with gradients.
    edge_align = _edge_alignment(mask01.astype(np.uint8), sig.edges01)

    # Shadow-like pixels in the boundary band are often background shadows wrongly included.
    core_k = max(5, int(round(0.012 * min(h, w))))
    core = cv2.erode(mask01.astype(np.uint8), _k(core_k), iterations=1)
    band_in = (mask01.astype(np.uint8) & (1 - core).astype(np.uint8)).astype(np.uint8)
    shadow = _shadow_like_mask_anywhere(sig, strictness=1.0)
    shadow_in = float(((shadow > 0) & (band_in > 0)).sum()) / float(band_in.sum() + 1e-6)

    # Fragmentation penalty (too many small components suggests texture/background).
    num_cc, _, stats, _ = cv2.connectedComponentsWithStats(mask01.astype(np.uint8), connectivity=8)
    small = int(np.sum((stats[1:, cv2.CC_STAT_AREA] < 80).astype(np.int32))) if num_cc > 1 else 0

    shape_score = 0.0
    shape_metrics: dict[str, float | int] = {"shape_solidity": 0.0, "shape_defects": 0, "shape_score": 0.0}
    if cfg is not None and bool(getattr(cfg, "shape_plausibility_enabled", True)):
        shape_score, shape_metrics = _shape_plausibility(mask01)

    score = 0.0
    score += 2.2 * float(extent)
    score += 1.0 * float(center_w)
    score += 1.0 * float(np.tanh(float(bg_sep) / 18.0))
    score += 1.25 * float(edge_align)
    score -= 2.9 * float(border_touch)
    score -= 1.15 * float(shadow_in)
    score -= 0.10 * float(max(0, num_cc - 2))
    score -= 0.06 * float(small)
    score += 0.15 * float(min(1.0, float(frac) / 0.35))
    if cfg is not None:
        w_shape = float(getattr(cfg, "shape_plausibility_weight", 0.08))
        score += float(max(0.0, w_shape)) * float(shape_score)

    metrics = {
        "area_frac": float(frac),
        "extent": float(extent),
        "border_touch": float(border_touch),
        "center_w": float(center_w),
        "bg_sep_u8": float(bg_sep),
        "edge_align": float(edge_align),
        "shadow_in": float(shadow_in),
        "num_cc": int(num_cc - 1),
        "small_cc": int(small),
        **{k: (float(v) if isinstance(v, (float, int)) else v) for k, v in shape_metrics.items()},
    }
    return float(score), metrics



# ── Type-specific candidate generators ──────────────────────────────────


def _latex_multiscale_edge_candidate(bgr: np.ndarray, sig: _SegSignals) -> np.ndarray:
    """
    Candidate for **latex** gloves (near-white on white background).

    Strategy: white top-hat enhances subtle luminance differences, then
    multi-scale Canny (σ=1.0, 2.0, 3.5) on both L and ab-gradient channels
    fuses edges at different scales.  The merged edge map is used as a barrier
    for flood-fill from the border → invert = foreground candidate.
    """
    h, w = bgr.shape[:2]
    gray = sig.lab_u8[:, :, 0]  # L channel

    # 1. White top-hat to enhance subtle contrast.
    th_k = max(21, int(round(min(h, w) * 0.06)) | 1)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (th_k, th_k)))
    # Boost so faint features become visible.
    enhanced = cv2.normalize(
        cv2.add(gray, tophat), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # 2. CLAHE on the enhanced L for more even gradients.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(enhanced)

    # 3. Multi-scale Canny on L channel.
    merged_edges = np.zeros((h, w), dtype=np.uint8)
    for sigma in (1.0, 2.0, 3.5):
        ks = int(round(sigma * 6)) | 1
        ks = max(3, ks)
        blurred = cv2.GaussianBlur(l_clahe, (ks, ks), sigma)
        edges = _auto_canny(blurred, sigma=0.30)
        merged_edges = cv2.max(merged_edges, edges)

    # 4. Multi-scale Canny on ab-gradient (shadow-invariant edges).
    a = cv2.GaussianBlur(sig.lab_u8[:, :, 1], (5, 5), 0)
    b = cv2.GaussianBlur(sig.lab_u8[:, :, 2], (5, 5), 0)
    ax = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    ay = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    bx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
    by = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
    mag_ab = cv2.magnitude(ax, ay) + cv2.magnitude(bx, by)
    mag_ab_u8 = cv2.normalize(mag_ab, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    for sigma in (1.5, 3.0):
        ks = int(round(sigma * 6)) | 1
        ks = max(3, ks)
        blurred_ab = cv2.GaussianBlur(mag_ab_u8, (ks, ks), sigma)
        edges_ab = _auto_canny(blurred_ab, sigma=0.28)
        merged_edges = cv2.max(merged_edges, edges_ab)

    # 5. Morphological external gradient as extra edge evidence.
    ext_grad = cv2.dilate(l_clahe, _k(3), iterations=1).astype(np.int16) - l_clahe.astype(np.int16)
    ext_grad = np.clip(ext_grad, 0, 255).astype(np.uint8)
    _, ext_edges = cv2.threshold(ext_grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    merged_edges = cv2.max(merged_edges, ext_edges)

    # 6. Build barrier from edges and flood-fill from border.
    dil = cv2.dilate(merged_edges, _k(3), iterations=1)
    kclose = max(9, int(round(min(h, w) * 0.018)))
    barrier = cv2.morphologyEx(dil, cv2.MORPH_CLOSE, _k(kclose), iterations=1)
    barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, _k(max(5, kclose // 2)), iterations=1)

    free = (barrier == 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=8)
    if num <= 1:
        return np.zeros((h, w), np.uint8)

    best_idx, best_score = -1, -1e9
    for i in range(1, num):
        x, y, ww, hh, area = stats[i].tolist()
        if area < int(0.02 * h * w):
            continue
        touches = (x <= 0) or (y <= 0) or (x + ww >= w) or (y + hh >= h)
        if touches:
            continue
        comp = (labels == i).astype(np.uint8)
        cw = float((sig.center_prior * comp.astype(np.float32)).sum()) / float(area + 1e-6)
        sc = float(area) * (0.7 + 0.9 * cw)
        if sc > best_score:
            best_score = sc
            best_idx = i

    if best_idx < 1:
        return np.zeros((h, w), np.uint8)

    cand = (labels == best_idx).astype(np.uint8)
    k_open = max(3, int(round(min(h, w) * 0.006)))
    k_close = max(7, int(round(min(h, w) * 0.015)))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, _k(k_open), iterations=1)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, _k(k_close), iterations=2)
    cand = _select_component_centered(cand)
    return cand.astype(np.uint8)


def _saturation_candidate(bgr: np.ndarray, sig: _SegSignals) -> np.ndarray:
    """
    Candidate for **leather** gloves using HSV saturation thresholding.

    Leather has high saturation (rich brown/tan colors) while the white background
    and cast shadows both have very low saturation.  Thresholding the S channel
    with an adaptive percentile-based threshold gives a strong initial mask.
    """
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]  # saturation channel

    # Border saturation should be low (white background).
    border_vals = s[sig.border_mask01 > 0]
    if border_vals.size < 80:
        return np.zeros((h, w), np.uint8)

    # Threshold: well above the border saturation.
    bg_s_p95 = float(np.percentile(border_vals, 95))
    # Also consider global distribution.
    global_p65 = float(np.percentile(s, 65))
    thr = max(bg_s_p95 + 15.0, global_p65)
    thr = int(min(220, max(25, thr)))

    fg = ((s >= thr) & (sig.center_prior > 0.04)).astype(np.uint8)

    k1 = max(3, int(round(min(h, w) * 0.006)))
    k2 = max(7, int(round(min(h, w) * 0.016)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _k(k1), iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _k(k2), iterations=2)
    fg = _select_component_centered(fg)

    if fg.sum() < int(0.03 * h * w):
        # Fallback: try a lower threshold.
        thr2 = int(max(15, bg_s_p95 + 8.0))
        fg = ((s >= thr2) & (sig.center_prior > 0.06)).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _k(k1), iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _k(k2), iterations=2)
        fg = _select_component_centered(fg)

    return fg.astype(np.uint8)


def _local_variance_candidate(bgr: np.ndarray, sig: _SegSignals) -> np.ndarray:
    """
    Candidate for **fabric** gloves using local intensity variance.

    Knit fabric has high local variance (repeating yarn pattern) while smooth
    white background has near-zero variance.  A sliding window variance map
    combined with the existing Laplacian texture energy provides a robust
    foreground mask.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Local variance via E[X²] - E[X]².
    ksize = max(9, int(round(min(h, w) * 0.025)) | 1)
    mean = cv2.blur(gray, (ksize, ksize))
    mean_sq = cv2.blur(gray * gray, (ksize, ksize))
    variance = np.clip(mean_sq - mean * mean, 0, None)
    var_u8 = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Border variance should be low (smooth background).
    border_vals = var_u8[sig.border_mask01 > 0]
    if border_vals.size < 80:
        return np.zeros((h, w), np.uint8)

    bg_var_p95 = float(np.percentile(border_vals, 95))
    global_p60 = float(np.percentile(var_u8, 60))
    thr = max(bg_var_p95 + 5.0, global_p60)
    thr = int(min(250, max(8, thr)))

    fg = ((var_u8 >= thr) & (sig.center_prior > 0.04)).astype(np.uint8)

    # Also combine with Laplacian texture energy for robustness.
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    tex_u8 = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    border_tex = tex_u8[sig.border_mask01 > 0]
    if border_tex.size >= 80:
        tex_thr = int(np.percentile(border_tex, 95) + 5.0)
        tex_thr = int(max(12, min(250, tex_thr)))
        tex_fg = ((tex_u8 >= tex_thr) & (sig.center_prior > 0.04)).astype(np.uint8)
        # Weighted OR: either strong variance or strong texture.
        fg = ((fg > 0) | (tex_fg > 0)).astype(np.uint8)

    # More aggressive closing for fuzzy fabric boundaries.
    k1 = max(3, int(round(min(h, w) * 0.005)))
    k2 = max(9, int(round(min(h, w) * 0.020)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _k(k1), iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _k(k2), iterations=2)
    fg = _select_component_centered(fg)
    return fg.astype(np.uint8)


def _chan_vese_candidate(
    bgr: np.ndarray,
    sig: _SegSignals,
    seed01: np.ndarray,
    cfg: SegmentationConfig,
) -> np.ndarray:
    """
    Optional region-based candidate for weak/soft boundaries (latex, fuzzy fabric).

    Uses Chan–Vese-style active contours (morphological variant when available).
    This is a *candidate generator* only; final selection still uses scoring + GrabCut.
    """
    h, w = bgr.shape[:2]
    max_side = int(max(128, getattr(cfg, "chan_vese_max_side", 512)))
    iters = int(max(10, getattr(cfg, "chan_vese_iters", 80)))
    smoothing = int(max(0, getattr(cfg, "chan_vese_smoothing", 2)))

    # Enhance subtle structure (especially latex-on-white) using the same idea as the
    # latex multiscale edge candidate: top-hat + CLAHE on L.
    l = sig.lab_u8[:, :, 0]
    th_k = max(21, int(round(min(h, w) * 0.06)) | 1)
    tophat = cv2.morphologyEx(
        l,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (th_k, th_k)),
    )
    enhanced = cv2.normalize(cv2.add(l, tophat), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_u8 = clahe.apply(enhanced)

    # Downscale for runtime stability.
    sf = float(min(1.0, float(max_side) / float(max(h, w) + 1e-6)))
    if sf < 0.999:
        hh = int(max(64, round(h * sf)))
        ww = int(max(64, round(w * sf)))
        img_u8_s = cv2.resize(img_u8, (ww, hh), interpolation=cv2.INTER_AREA)
        seed_s = cv2.resize((seed01 > 0).astype(np.uint8), (ww, hh), interpolation=cv2.INTER_NEAREST)
    else:
        img_u8_s = img_u8
        seed_s = (seed01 > 0).astype(np.uint8)

    # Seed must not be empty; fallback to a centered disk.
    if int(seed_s.sum()) < int(0.004 * seed_s.size):
        seed_s = np.zeros_like(seed_s, dtype=np.uint8)
        r = int(round(0.22 * float(min(seed_s.shape[:2]))))
        r = int(max(12, r))
        cv2.circle(seed_s, (seed_s.shape[1] // 2, seed_s.shape[0] // 2), r, 1, -1)

    img_f = (img_u8_s.astype(np.float32) / 255.0).astype(np.float32)
    init = (seed_s > 0)

    seg01_s: np.ndarray | None = None
    try:
        from skimage.segmentation import morphological_chan_vese  # type: ignore

        seg = morphological_chan_vese(
            img_f,
            num_iter=int(iters),
            init_level_set=init,
            smoothing=int(smoothing),
        )
        seg01_s = (seg > 0).astype(np.uint8)
    except Exception:
        # Fallback: classic Chan–Vese implementation.
        try:
            from skimage.segmentation import chan_vese  # type: ignore

            seg = chan_vese(
                img_f,
                mu=0.25,
                lambda1=1.0,
                lambda2=1.0,
                tol=1e-3,
                max_num_iter=int(iters),
                dt=0.5,
                init_level_set=init,
                extended_output=False,
            )
            seg01_s = (seg > 0).astype(np.uint8)
        except Exception:
            return np.zeros((h, w), np.uint8)

    # Pick polarity by overlap with the seed.
    seed01_s = (seed_s > 0).astype(np.uint8)
    iou_a = _mask_iou(seg01_s, seed01_s)
    inv = (1 - seg01_s).astype(np.uint8)
    iou_b = _mask_iou(inv, seed01_s)
    if iou_b > iou_a + 0.01:
        seg01_s = inv

    if sf < 0.999:
        seg01 = cv2.resize(seg01_s, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        seg01 = seg01_s

    # Mild cleanup: keep centered main component and close small gaps.
    k1 = max(3, int(round(min(h, w) * 0.006)))
    k2 = max(9, int(round(min(h, w) * 0.018)))
    seg01 = cv2.morphologyEx(seg01.astype(np.uint8), cv2.MORPH_OPEN, _k(k1), iterations=1)
    seg01 = cv2.morphologyEx(seg01.astype(np.uint8), cv2.MORPH_CLOSE, _k(k2), iterations=2)
    seg01 = _select_component_centered(_largest_component(seg01))
    return seg01.astype(np.uint8)


def _prepare_candidates(
    bgr: np.ndarray,
    sig: _SegSignals,
    cfg: SegmentationConfig,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[tuple[str, np.ndarray, float, dict]], dict[str, dict[str, float | bool]]]:
    h, w = bgr.shape[:2]
    cand_bg, _sure_bg = _bg_distance_candidate(sig)
    cand_km = _kmeans_segment(bgr, k=3)
    l = sig.lab_u8[:, :, 0]
    cand_hi, cand_lo = _threshold_segment_lab_l_bidirectional(l)
    cand_chroma = _chroma_candidate(sig)
    cand_texture = _texture_candidate(bgr, sig)
    cand_edge = _edge_closed_candidate(sig)
    cand_edge_s = _edge_closed_candidate_strong(sig)

    k_open = max(3, int(round(min(h, w) * 0.008)))
    k_close = max(9, int(round(min(h, w) * 0.022)))
    cand_hi = cv2.morphologyEx(cand_hi, cv2.MORPH_OPEN, _k(k_open), iterations=1)
    cand_hi = cv2.morphologyEx(cand_hi, cv2.MORPH_CLOSE, _k(k_close), iterations=2)
    cand_hi = _select_component_centered(_largest_component(cand_hi))
    cand_lo = cv2.morphologyEx(cand_lo, cv2.MORPH_OPEN, _k(k_open), iterations=1)
    cand_lo = cv2.morphologyEx(cand_lo, cv2.MORPH_CLOSE, _k(k_close), iterations=2)
    cand_lo = _select_component_centered(_largest_component(cand_lo))

    candidates = {
        "bg_dist": cand_bg,
        "kmeans": cand_km,
        "labL_hi": cand_hi,
        "labL_lo": cand_lo,
        "chroma_fg": cand_chroma,
        "texture_fg": cand_texture,
        "edge_closed": cand_edge,
        "edge_closed_strong": cand_edge_s,
    }

    # ── Type-specific candidates (conditionally enabled) ────────────────
    if bool(cfg.latex_edge_multiscale):
        candidates["latex_ms_edge"] = _latex_multiscale_edge_candidate(bgr, sig)
    if bool(cfg.leather_saturation_candidate):
        candidates["saturation_fg"] = _saturation_candidate(bgr, sig)
    if bool(cfg.fabric_variance_candidate):
        candidates["variance_fg"] = _local_variance_candidate(bgr, sig)

    silhouette_ref = _build_silhouette_ref(cand_lo, cand_edge, cand_edge_s, h, w)
    profile = str(cfg.profile_name).strip().lower()

    # Build boost/suppress dicts from config tuples.
    boost_map: dict[str, float] = {str(k): float(v) for k, v in cfg.candidate_boost} if cfg.candidate_boost else {}
    suppress_map: dict[str, float] = {str(k): float(v) for k, v in cfg.candidate_suppress} if cfg.candidate_suppress else {}

    scored: list[tuple[str, np.ndarray, float, dict]] = []
    validity: dict[str, dict[str, float | bool]] = {}
    for m, mk in candidates.items():
        area_frac = _mask_area_frac(mk)
        iou_sil = _mask_iou(mk, silhouette_ref) if silhouette_ref.sum() > 0 else 0.0
        valid = True
        min_area = 0.0
        min_iou = 0.0
        if m == "chroma_fg":
            min_area = float(cfg.chroma_min_area_frac)
            min_iou = float(cfg.chroma_min_iou_silhouette)
            if profile == "leather":
                min_area *= 0.55
                min_iou *= 0.45
            if area_frac < min_area or iou_sil < min_iou:
                valid = False
        sc, metrics = _score_mask(bgr, mk, sig, cfg=cfg)
        if sc > -1e8:
            # Legacy profile-based adjustments (kept for backward compat).
            if profile in {"latex", "fabric"} and m in {"labL_lo", "edge_closed", "edge_closed_strong"}:
                sc += 0.04
            if m == "chroma_fg":
                if profile == "leather" and valid:
                    sc += 0.06
                elif profile in {"latex", "fabric"} and valid:
                    sc -= 0.03
            # ── Configurable per-candidate boost/suppress ────────────
            if m in boost_map:
                sc += boost_map[m]
            if m in suppress_map:
                sc -= suppress_map[m]
        if not valid:
            sc = -1e9
            metrics = {
                **(metrics if isinstance(metrics, dict) else {}),
                "reason": "chroma_invalid",
            }
        scored.append((m, mk, float(sc), metrics if isinstance(metrics, dict) else {}))
        validity[m] = {
            "valid": bool(valid),
            "area_frac": float(area_frac),
            "iou_silhouette": float(iou_sil),
            "min_area_frac": float(min_area),
            "min_iou_silhouette": float(min_iou),
        }

    scored.sort(key=lambda x: float(x[2]), reverse=True)

    # ── Optional region-based fallback candidate (Chan–Vese) ─────────────
    if bool(cfg.chan_vese_enabled) and profile in {"latex", "fabric"} and scored:
        top_metrics = scored[0][3] if isinstance(scored[0][3], dict) else {}
        top_bg_sep = float(top_metrics.get("bg_sep_u8", 0.0))
        top_edge_align = float(top_metrics.get("edge_align", 0.0))
        low_conf = bool(
            (top_bg_sep < float(cfg.chan_vese_low_conf_bg_sep))
            and (top_edge_align < float(cfg.chan_vese_low_conf_edge_align))
        )
        run_cv = (not bool(cfg.chan_vese_only_if_low_confidence)) or low_conf
        if run_cv:
            seed = silhouette_ref if int(silhouette_ref.sum()) > 0 else scored[0][1]
            cand_cv = _chan_vese_candidate(bgr, sig, seed, cfg)
            m = "chan_vese_fg"
            candidates[m] = cand_cv
            area_frac = _mask_area_frac(cand_cv)
            iou_sil = _mask_iou(cand_cv, silhouette_ref) if silhouette_ref.sum() > 0 else 0.0
            sc, metrics = _score_mask(bgr, cand_cv, sig, cfg=cfg)
            if sc > -1e8:
                if m in boost_map:
                    sc += boost_map[m]
                if m in suppress_map:
                    sc -= suppress_map[m]
            scored.append((m, cand_cv, float(sc), metrics if isinstance(metrics, dict) else {}))
            validity[m] = {
                "valid": True,
                "area_frac": float(area_frac),
                "iou_silhouette": float(iou_sil),
                "min_area_frac": 0.0,
                "min_iou_silhouette": 0.0,
            }
            scored.sort(key=lambda x: float(x[2]), reverse=True)

    return candidates, silhouette_ref, scored, validity


def _grabcut_refine(
    bgr: np.ndarray,
    init_mask01: np.ndarray,
    aux_mask01: np.ndarray | None = None,
    sure_bg_mask01: np.ndarray | None = None,
    cfg: SegmentationConfig | None = None,
    runtime: dict[str, object] | None = None,
    sig: _SegSignals | None = None,
    use_trimap: bool = False,
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
    inter_min_ratio = float(getattr(cfg, "grabcut_inter_min_ratio", 0.78)) if cfg is not None else 0.78
    inter_ratio = float(inter.sum()) / float(init.sum() + 1e-6)
    # If aux misses significant parts of init (commonly fingertip pixels), do not
    # let the sure-FG core collapse to the intersection.
    if (inter.sum() < 0.004 * h * w) or (inter_ratio < inter_min_ratio):
        inter = init.copy()

    # Tip protection: keep fingertip pixels from the initial mask even if GrabCut
    # peels them off due to low contrast or seeding uncertainty.
    tip_keep = np.zeros((h, w), dtype=np.uint8)
    if cfg is None or bool(getattr(cfg, "tip_protect_enabled", True)):
        frac = float(getattr(cfg, "tip_protect_frac", 0.18)) if cfg is not None else 0.18
        kd = float(getattr(cfg, "tip_protect_dilate_frac", 0.006)) if cfg is not None else 0.006
        tip_keep = _tip_keep_mask_upright(bgr, init, frac=frac, dilate_frac=kd)
        tip_keep = (tip_keep & init).astype(np.uint8)

    # Configurable seeding geometry.
    erode_frac = float(cfg.grabcut_erode_frac) if cfg is not None else 0.012
    dilate_frac = float(cfg.grabcut_dilate_frac) if cfg is not None else 0.020
    sure_bg_frac = float(cfg.grabcut_sure_bg_dilate_frac) if cfg is not None else 0.09
    erode_sz = max(3, int(round(min(h, w) * erode_frac)))
    dilate_sz = max(5, int(round(min(h, w) * dilate_frac)))
    sure_fg = cv2.erode(inter, _k(erode_sz), iterations=1)
    prob_fg = cv2.dilate(union, _k(dilate_sz), iterations=1)
    sure_bg = (1 - cv2.dilate(prob_fg, _k(max(11, int(round(min(h, w) * sure_bg_frac)))), iterations=1)).astype(np.uint8)

    # Optional tri-map seeding based on background-distance (multi-level Otsu).
    # This provides a more principled initialization than pure morphology on the seed mask,
    # especially for near-white latex on white backgrounds.
    trimap_stats: dict[str, float | bool | int | str] = {"enabled": bool(use_trimap), "applied": False}
    tri_low: np.ndarray | None = None
    tri_high: np.ndarray | None = None
    tri_seed_fg: np.ndarray | None = None
    if bool(use_trimap) and (sig is not None):
        try:
            d = sig.d_bg.astype(np.float32)
            border = sig.border_mask01 > 0
            border_vals = d[border]
            if border_vals.size >= 80:
                lo = float(np.percentile(border_vals, 2.0))
            else:
                lo = float(np.percentile(d, 2.0))
            hi = float(np.percentile(d, 99.5))
            if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo + 1e-6):
                d8 = sig.d_bg_u8.copy()
                trimap_stats["robust_scale"] = False
            else:
                d8f = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
                d8 = (d8f * 255.0).astype(np.uint8)
                trimap_stats["robust_scale"] = True
                trimap_stats["scale_lo"] = float(lo)
                trimap_stats["scale_hi"] = float(hi)

            classes = int(getattr(cfg, "grabcut_trimap_classes", 3)) if cfg is not None else 3
            classes = int(max(3, min(5, classes)))
            thr: list[int] = []
            try:
                from skimage.filters import threshold_multiotsu  # type: ignore

                thr = [int(x) for x in np.ravel(threshold_multiotsu(d8, classes=classes)).tolist()]
                thr = sorted(thr)
            except Exception:
                thr = []

            if len(thr) >= 2:
                tri_low = (d8 <= int(thr[0])).astype(np.uint8)
                tri_high = (d8 > int(thr[-1])).astype(np.uint8)

                # Restrict "high" seed to be near the candidate region to reduce leakage.
                cap = cv2.dilate(union, _k(max(11, int(round(min(h, w) * 0.06)))), iterations=1)
                tri_seed_fg = (tri_high & (cap > 0).astype(np.uint8)).astype(np.uint8)

                # Derive a stricter sure-FG from the intersection between an eroded core
                # and the high tri-map class. Fall back if it becomes too small.
                sure_fg2 = (sure_fg & tri_seed_fg).astype(np.uint8)
                if int(sure_fg2.sum()) >= 80:
                    sure_fg = sure_fg2

                # Expand probable FG slightly using the tri-map "high" region (capped).
                prob_fg = ((prob_fg > 0) | (tri_seed_fg > 0)).astype(np.uint8)
                sure_bg = (1 - cv2.dilate(prob_fg, _k(max(11, int(round(min(h, w) * sure_bg_frac)))), iterations=1)).astype(np.uint8)

                trimap_stats["applied"] = True
                trimap_stats["classes"] = int(classes)
                trimap_stats["thr0_u8"] = int(thr[0])
                trimap_stats["thr_last_u8"] = int(thr[-1])
                trimap_stats["low_frac"] = float(tri_low.sum()) / float(h * w + 1e-6)
                trimap_stats["high_frac"] = float(tri_high.sum()) / float(h * w + 1e-6)
                trimap_stats["seed_high_frac"] = float(tri_seed_fg.sum()) / float(h * w + 1e-6)
            else:
                trimap_stats["reason"] = "multiotsu_failed"
        except Exception as e:
            trimap_stats["reason"] = f"exception:{type(e).__name__}"

    if runtime is not None and bool(use_trimap):
        runtime["grabcut_trimap"] = dict(trimap_stats)

    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    if tri_low is not None and tri_high is not None:
        # Seed background/foreground priors, but allow the geometric candidate seeds
        # below to override (latex can be low-contrast).
        mask[tri_low > 0] = cv2.GC_BGD
        if tri_seed_fg is not None:
            mask[tri_seed_fg > 0] = cv2.GC_PR_FGD
        else:
            mask[tri_high > 0] = cv2.GC_PR_FGD

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

    # Candidate-based expansion cap: prevents GrabCut from filling areas the
    # candidate didn't cover (e.g. inter-finger gaps in fabric).
    cap_frac = float(cfg.grabcut_candidate_cap_frac) if cfg is not None else 0.0
    if cap_frac > 0.0:
        cap_k = max(5, int(round(min(h, w) * cap_frac)))
        cand_cap = cv2.dilate(union, _k(cap_k), iterations=1)
        out01 = (out01 & cand_cap).astype(np.uint8)

    # Only cap expansion when we see evidence of leakage; unconditional caps can trim fine edges.
    cap_thr = float(cfg.grabcut_cap_border_touch) if cfg is not None else 0.10
    if _border_touch_ratio(out01) > cap_thr:
        expand_cap = cv2.dilate(union, _k(max(13, int(round(min(h, w) * 0.085)))), iterations=1)
        out01 &= expand_cap

    # Opening can trim fine glove edges (fabric/latex). Keep it small unless we see leakage/noise.
    bt_pre = _border_touch_ratio(out01)
    if cfg is None:
        k_open = 3 if bt_pre < 0.04 else max(3, int(round(min(h, w) * 0.004)))
        k_close = min(9, int(max(5, int(round(min(h, w) * 0.006)))))
    else:
        k_open = int(max(1, cfg.grabcut_open_k))
        k_close = int(max(1, cfg.grabcut_close_k))
    out01 = cv2.morphologyEx(out01, cv2.MORPH_OPEN, _k(k_open), iterations=1)
    out01 = cv2.morphologyEx(out01, cv2.MORPH_CLOSE, _k(k_close), iterations=1)
    out01 = _select_component_centered(out01)
    # Boundary color pruning can be overly conservative on white backgrounds (e.g., latex highlights).
    # Apply it only when we see mild leakage risk.
    prune_stats: dict[str, float | bool | str] = {}
    if _border_touch_ratio(out01) > 0.03:
        out01 = _prune_boundary_color_outliers(bgr, out01, cfg=cfg, stats=prune_stats)
    if runtime is not None and prune_stats:
        lst = runtime.setdefault("prune_stats", [])
        if isinstance(lst, list):
            lst.append({"stage": "grabcut", **prune_stats})
    bt = _border_touch_ratio(out01)
    if bt > 0.11:
        trimmed = _disconnect_border_leaks(out01)
        if _border_touch_ratio(trimmed) + 0.02 < bt:
            out01 = trimmed
    if int(tip_keep.sum()) > 0:
        out01 = ((out01 > 0) | (tip_keep > 0)).astype(np.uint8)
        out01 = _select_component_centered(out01)
    return out01


def _recover_trimmed_edges(
    bgr: np.ndarray,
    mask01: np.ndarray,
    sig: _SegSignals,
    roi01: np.ndarray | None = None,
    cfg: SegmentationConfig | None = None,
    runtime: dict[str, object] | None = None,
    *,
    kd_scale: float = 1.0,
    ring_scale: float = 1.0,
    max_area_mul: float = 1.22,
) -> np.ndarray:
    """
    Attempt to recover boundary pixels that got trimmed by conservative cleanup.

    We grow the mask slightly into a narrow ring using a hysteresis rule:
    - allow growth into pixels that are sufficiently different from background (d_bg),
      and/or supported by strong edges near the current boundary.

    This is conservative by construction and will revert if it increases border leakage.
    """
    m = (mask01 > 0).astype(np.uint8)
    h, w = m.shape[:2]
    area0 = float(m.sum())
    if area0 < 120:
        return m
    roi = np.ones((h, w), np.uint8) if roi01 is None else (roi01 > 0).astype(np.uint8)

    # Stage 1: small dilation then prune to avoid pulling in shadows/background.
    kd = max(3, int(round(float(kd_scale) * min(h, w) * 0.004)))
    cand = cv2.dilate(m, _k(kd), iterations=1).astype(np.uint8)
    cand = (m | (cand & roi)).astype(np.uint8)
    cand = _select_component_centered(cand)
    if _border_touch_ratio(cand) <= max(0.12, _border_touch_ratio(m) + 0.04):
        prune_stats: dict[str, float | bool | str] = {}
        cand2 = _prune_boundary_color_outliers(bgr, cand, cfg=cfg, sig=sig, stats=prune_stats)
        if runtime is not None and prune_stats:
            lst = runtime.setdefault("prune_stats", [])
            if isinstance(lst, list):
                lst.append({"stage": "edge_recover", **prune_stats})
        cand2 = _select_component_centered(cand2)
        if cand2.sum() > m.sum() and cand2.sum() <= int(area0 * min(1.18, float(max_area_mul))):
            m = cand2
            area0 = float(m.sum())

    # Stage 2: grow into a narrow band with hysteresis on background-distance + edge support.
    r = max(7, int(round(float(ring_scale) * min(h, w) * 0.015)))
    near = cv2.dilate(m, _k(r), iterations=1)
    ring = (near > 0).astype(np.uint8) & (m == 0).astype(np.uint8)
    ring &= roi
    if ring.sum() < 50:
        return m

    border_vals = sig.d_bg_u8[sig.border_mask01 > 0]
    if border_vals.size >= 80:
        p99 = float(np.percentile(border_vals, 99))
        p90 = float(np.percentile(sig.d_bg_u8, 90))
        thr_high = float(min(254.0, max(p99 + 10.0, p90)))
        thr_low = float(min(252.0, max(p99 + 4.0, (thr_high - 14.0))))
        edge_thr = float(max(0.0, p99 - 2.0))
    else:
        thr_high = float(np.percentile(sig.d_bg_u8, 92))
        thr_low = float(np.percentile(sig.d_bg_u8, 80))
        edge_thr = float(np.percentile(sig.d_bg_u8, 70))

    d8 = sig.d_bg_u8.astype(np.float32)
    edges = sig.edges01.astype(np.uint8)

    # Prefer edge-supported growth; d_bg alone can pull in soft shadows on white backgrounds.
    strong = (((edges > 0) & (d8 >= edge_thr)) & (ring > 0)).astype(np.uint8)
    allow = (((d8 >= thr_low) | (strong > 0)) & (ring > 0)).astype(np.uint8)

    if strong.sum() < 25:
        # Nothing convincing to add.
        return m

    grow = (m | strong).astype(np.uint8)
    k3 = _k(3)
    for _ in range(40):
        nxt = (grow | (cv2.dilate(grow, k3, iterations=1) & allow)).astype(np.uint8)
        if int(nxt.sum()) == int(grow.sum()):
            break
        grow = nxt

    # Sanity checks to avoid introducing leakage.
    bt0 = _border_touch_ratio(m)
    bt1 = _border_touch_ratio(grow)
    if bt1 > max(0.16, bt0 + 0.05):
        return m

    # Ensure we don't blow up the area.
    area1 = float(grow.sum())
    if area1 > area0 * float(max_area_mul):
        return m

    # Keep the best centered component and gently close small gaps.
    grow = _select_component_centered(grow)
    grow = cv2.morphologyEx(grow, cv2.MORPH_CLOSE, _k(5), iterations=1)
    return grow.astype(np.uint8)


def _bg_distance_candidate(sig: _SegSignals) -> tuple[np.ndarray, np.ndarray]:
    """
    Candidate from robust background distance thresholding.

    Returns:
    - foreground candidate mask01
    - sure background mask01 (for seeding GrabCut)
    """
    h, w = sig.d_bg_u8.shape[:2]
    d8 = sig.d_bg_u8

    border_vals = d8[sig.border_mask01 > 0]
    if border_vals.size < 80:
        thr = int(np.percentile(d8, 90))
        bg_thr = int(np.percentile(d8, 60))
    else:
        border_p99 = int(np.percentile(border_vals, 99))
        border_p95 = int(np.percentile(border_vals, 95))
        bg_thr = int(np.percentile(border_vals, 75))

        # Otsu can be unstable if the histogram is near-flat; still use it as a weak cue.
        otsu_thr, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_thr = int(otsu_thr)

        p90 = int(np.percentile(d8, 90))
        thr = max(otsu_thr, border_p99 + 6, p90)
        thr = min(252, max(thr, border_p95 + 8))

    fg = (d8 >= int(thr)).astype(np.uint8)
    fg = (fg & (sig.center_prior > 0.06)).astype(np.uint8)

    k1 = max(3, int(round(min(h, w) * 0.007)))
    k2 = max(7, int(round(min(h, w) * 0.016)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _k(k1), iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _k(k2), iterations=2)
    fg = _select_component_centered(fg)
    if fg.sum() < int(0.01 * h * w):
        fg = ((sig.center_prior > 0.38) & (d8 >= int(np.percentile(d8, 65)))).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _k(k2), iterations=1)
        fg = _select_component_centered(fg)

    sure_bg = ((d8 <= int(bg_thr)) & (sig.center_prior < 0.78)).astype(np.uint8)
    sure_bg |= sig.border_mask01.astype(np.uint8)
    return fg.astype(np.uint8), sure_bg.astype(np.uint8)


def _edge_closed_candidate(sig: _SegSignals) -> np.ndarray:
    """
    Candidate from edge-closed regions: treat edges as barriers and select an interior
    region that does NOT touch the image border.
    """
    h, w = sig.edges01.shape[:2]
    edges = sig.edges01.astype(np.uint8)
    if edges.sum() < int(0.002 * h * w):
        return np.zeros((h, w), np.uint8)

    dil = cv2.dilate(edges, _k(3), iterations=1)
    kclose = max(7, int(round(min(h, w) * 0.014)))
    barrier = cv2.morphologyEx(dil, cv2.MORPH_CLOSE, _k(kclose), iterations=1)
    barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, _k(max(5, kclose // 2)), iterations=1)

    free = (barrier == 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=8)
    if num <= 1:
        return np.zeros((h, w), np.uint8)

    best_idx = -1
    best_score = -1e9
    for i in range(1, num):
        x, y, ww, hh, area = stats[i].tolist()
        if area < int(0.02 * h * w):
            continue
        touches_border = (x <= 0) or (y <= 0) or (x + ww >= w) or (y + hh >= h)
        if touches_border:
            continue
        comp = (labels == i).astype(np.uint8)
        cw = float((sig.center_prior * comp.astype(np.float32)).sum()) / float(area + 1e-6)
        score = float(area) * (0.7 + 0.9 * cw)
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx < 1:
        return np.zeros((h, w), np.uint8)

    cand = (labels == best_idx).astype(np.uint8)
    kand = max(5, int(round(min(h, w) * 0.012)))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, _k(max(3, kand // 2)), iterations=1)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, _k(kand), iterations=2)
    cand = _select_component_centered(cand)
    return cand.astype(np.uint8)


def _edge_closed_candidate_strong(sig: _SegSignals) -> np.ndarray:
    """
    Stronger edge barrier closing for cases where edges are slightly broken.
    More likely to fill narrow gaps (thumb-index webbing), so it is treated as
    an additional candidate instead of replacing the default.
    """
    h, w = sig.edges01.shape[:2]
    edges = sig.edges01.astype(np.uint8)
    if edges.sum() < int(0.002 * h * w):
        return np.zeros((h, w), np.uint8)

    dil = cv2.dilate(edges, _k(3), iterations=1)
    kclose = max(11, int(round(min(h, w) * 0.028)))
    barrier = cv2.morphologyEx(dil, cv2.MORPH_CLOSE, _k(kclose), iterations=1)
    barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, _k(max(7, kclose // 2)), iterations=1)

    free = (barrier == 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=8)
    if num <= 1:
        return np.zeros((h, w), np.uint8)

    best_idx = -1
    best_score = -1e9
    for i in range(1, num):
        x, y, ww, hh, area = stats[i].tolist()
        if area < int(0.02 * h * w):
            continue
        touches_border = (x <= 0) or (y <= 0) or (x + ww >= w) or (y + hh >= h)
        if touches_border:
            continue
        comp = (labels == i).astype(np.uint8)
        cw = float((sig.center_prior * comp.astype(np.float32)).sum()) / float(area + 1e-6)
        score = float(area) * (0.7 + 0.9 * cw)
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx < 1:
        return np.zeros((h, w), np.uint8)

    cand = (labels == best_idx).astype(np.uint8)
    kand = max(5, int(round(min(h, w) * 0.012)))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, _k(max(3, kand // 2)), iterations=1)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, _k(kand), iterations=2)
    cand = _select_component_centered(cand)
    return cand.astype(np.uint8)


def _watershed_refine(
    bgr: np.ndarray,
    init_mask01: np.ndarray,
    sig: _SegSignals,
    cfg: SegmentationConfig | None = None,
    runtime: dict[str, object] | None = None,
) -> np.ndarray:
    h, w = init_mask01.shape[:2]
    init = (init_mask01 > 0).astype(np.uint8)
    if init.sum() < int(0.01 * h * w):
        return init

    erode_sz = max(3, int(round(min(h, w) * 0.015)))
    sure_fg = cv2.erode(init, _k(erode_sz), iterations=1)
    if sure_fg.sum() < 80:
        sure_fg = np.zeros((h, w), np.uint8)
        r = max(10, int(round(min(h, w) * 0.05)))
        cv2.circle(sure_fg, (w // 2, h // 2), r, 1, -1)

    markers = np.zeros((h, w), np.int32)
    markers[sig.border_mask01 > 0] = 1
    markers[sure_fg > 0] = 2

    # Watershed expects a 3-channel image; use the original BGR.
    markers = cv2.watershed(bgr, markers)
    out01 = (markers == 2).astype(np.uint8)

    # Constrain with a soft expansion cap around the seed.
    cap = cv2.dilate(init, _k(max(9, int(round(min(h, w) * 0.06)))), iterations=1)
    out01 &= cap

    out01 = cv2.morphologyEx(out01, cv2.MORPH_OPEN, _k(5), iterations=1)
    out01 = cv2.morphologyEx(out01, cv2.MORPH_CLOSE, _k(7), iterations=1)
    out01 = _select_component_centered(out01)
    prune_stats: dict[str, float | bool | str] = {}
    if _border_touch_ratio(out01) > 0.03:
        out01 = _prune_boundary_color_outliers(bgr, out01, cfg=cfg, sig=sig, stats=prune_stats)
    if runtime is not None and prune_stats:
        lst = runtime.setdefault("prune_stats", [])
        if isinstance(lst, list):
            lst.append({"stage": "watershed", **prune_stats})
    bt = _border_touch_ratio(out01)
    if bt > 0.11:
        trimmed = _disconnect_border_leaks(out01)
        if _border_touch_ratio(trimmed) + 0.02 < bt:
            out01 = trimmed
    return out01.astype(np.uint8)


def _upright_finger_refine(
    bgr: np.ndarray,
    mask01: np.ndarray,
    cfg: SegmentationConfig,
    runtime: dict[str, object] | None = None,
) -> np.ndarray:
    """
    Rotate the glove upright (PCA) and apply stronger edge recovery only over the
    finger region (top part of the upright glove), then rotate back.

    This helps recover fingertips/edges that are often trimmed on fabric/latex
    without bloating the cuff/palm region.
    """
    m0 = (mask01 > 0).astype(np.uint8)
    if m0.sum() < 400:
        return m0

    rot_bgr, rot_m01, mat = _upright_normalize(bgr, m0)

    roi, (x, y, ww, hh) = _finger_roi_from_mask(rot_m01, frac=float(cfg.upright_roi_frac))
    if hh < 60 or ww < 60:
        inv = cv2.invertAffineTransform(mat)
        back = cv2.warpAffine(rot_m01, inv, (m0.shape[1], m0.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)
        return _select_component_centered((back > 0).astype(np.uint8))

    # Tip protection in the upright frame: keep a fingertip-end band from the
    # incoming mask to avoid accidental tip trimming by valley/webbing carving.
    tip_keep = np.zeros_like(rot_m01, dtype=np.uint8)
    if bool(getattr(cfg, "tip_protect_enabled", True)):
        xT, yT, wwT, hhT = cv2.boundingRect(rot_m01.astype(np.uint8))
        if wwT > 0 and hhT > 0:
            f = float(max(0.06, min(0.35, float(getattr(cfg, "tip_protect_frac", 0.18)))))
            band_h = int(max(6, round(f * float(hhT))))
            y_end_top = int(min(rot_m01.shape[0], max(0, yT + band_h)))
            y_start_bot = int(max(0, min(rot_m01.shape[0], (yT + hhT) - band_h)))
            tip_keep[yT:y_end_top, xT : xT + wwT] = 1
            tip_keep[y_start_bot : yT + hhT, xT : xT + wwT] = 1
            tip_keep = (tip_keep & (rot_m01 > 0).astype(np.uint8)).astype(np.uint8)
            if int(tip_keep.sum()) > 0:
                kd = max(3, int(round(float(getattr(cfg, "tip_protect_dilate_frac", 0.006)) * float(min(rot_m01.shape[:2])))))
                kd = int(min(19, max(3, kd | 1)))
                tip_keep = cv2.dilate(tip_keep, _k(kd), iterations=1).astype(np.uint8)

    sig = _compute_signals(rot_bgr)
    shadow_any = _shadow_like_mask_anywhere(sig, strictness=float(cfg.shadow_strictness))
    refined = _recover_trimmed_edges(
        rot_bgr,
        rot_m01,
        sig,
        roi01=roi,
        cfg=cfg,
        runtime=runtime,
        kd_scale=float(cfg.upright_kd_scale),
        ring_scale=float(cfg.upright_ring_scale),
        max_area_mul=float(cfg.upright_max_area_mul),
    )

    # Carve out mistakenly filled valleys (e.g., thumb-index webbing, finger gaps) using a glove-aware prior:
    # only carve within the finger ROI, and only within a boundary band (never the eroded core).
    refined = (refined > 0).astype(np.uint8)
    core_k = max(5, int(round(0.012 * min(refined.shape[:2]))))
    core = cv2.erode(refined, _k(core_k), iterations=1)
    boundary_inside = (refined & (1 - core)).astype(np.uint8)

    if sig.d_bg_u8[sig.border_mask01 > 0].size >= 80:
        border_vals = sig.d_bg_u8[sig.border_mask01 > 0]
        bg_like_thr = float(np.percentile(border_vals, float(cfg.bg_like_percentile)) + float(cfg.bg_like_margin))
        bg_like_thr = float(max(6.0, min(80.0, bg_like_thr)))
    else:
        bg_like_thr = _bg_like_thr_u8(sig)

    x2, y2, ww2, hh2 = cv2.boundingRect(refined.astype(np.uint8))
    x0 = int(x2)
    x1 = int(x2 + ww2)
    y0 = int(y2)
    y1 = int(y2 + hh2)
    y_top_band = int(round(y0 + 0.48 * float(hh2)))

    if ww2 >= 50 and hh2 >= 80:
        # Top profile: for each x, find first foreground y (within the bbox).
        prof = np.full((ww2,), y_top_band, dtype=np.float32)
        for xi in range(x0, x1):
            col = refined[y0:y_top_band, xi]
            ys = np.where(col > 0)[0]
            if ys.size:
                prof[xi - x0] = float(y0 + int(ys[0]))

        # Smooth profile to reduce texture noise (fabric).
        win = max(9, int(round(ww2 * 0.06)))
        if win % 2 == 0:
            win += 1
        win = min(win, 51)
        if win >= 9:
            k = np.ones((win,), np.float32) / float(win)
            prof_s = np.convolve(prof, k, mode="same")
        else:
            prof_s = prof

        # Valleys are local maxima of prof (deeper downward between fingers/thumb).
        prom_thr = float(max(10.0, float(cfg.valley_prom_frac) * float(hh2)))
        valleys: list[tuple[float, int]] = []
        for i in range(3, ww2 - 3):
            if not (prof_s[i - 1] < prof_s[i] > prof_s[i + 1]):
                continue
            left = float(np.min(prof_s[max(0, i - 8) : i]))
            right = float(np.min(prof_s[i + 1 : min(ww2, i + 9)]))
            prom = float(prof_s[i] - max(left, right))
            if prom < prom_thr:
                continue
            # Skip extreme ends of the glove bbox.
            if i < int(0.10 * ww2) or i > int(0.90 * ww2):
                continue
            valleys.append((prom, i))

        valleys.sort(reverse=True, key=lambda t: float(t[0]))
        valleys = valleys[: int(max(1, cfg.valley_max_count))]

        notch_w = int(max(12, round(float(cfg.valley_notch_w_frac) * ww2)))
        notch_depth = int(max(16, round(float(cfg.valley_notch_depth_frac) * hh2)))
        notch_depth = min(notch_depth, int(max(20, round(float(cfg.valley_notch_depth_cap_frac) * hh2))))

        carve = np.zeros_like(refined, dtype=np.uint8)
        for _prom, i in valleys:
            xv = int(x0 + i)
            yv = int(round(float(prof[i])))
            if yv >= y_top_band - 2:
                continue
            xL = max(x0, xv - notch_w // 2)
            xR = min(x1, xv + notch_w // 2 + 1)
            yA = max(y0, yv - 2)
            yB = min(y_top_band, yv + notch_depth)
            if yB <= yA + 4:
                continue
            carve[yA:yB, xL:xR] = 1

        if carve.sum() > 0 and bool(cfg.valley_carve_enabled):
            # Only carve background-like pixels, and only in the boundary band.
            dt = cv2.distanceTransform((refined * 255).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
            thin = dt <= float(max(7.0, float(cfg.thin_dt_frac) * float(min(hh2, ww2))))
            rm = (
                (carve > 0)
                & (roi > 0)
                & (boundary_inside > 0)
                & (thin > 0)
                & ((sig.d_bg_u8.astype(np.float32) <= float(bg_like_thr)) | (shadow_any > 0))
            )
            if int(tip_keep.sum()) > 0:
                rm = rm & (tip_keep == 0)
            refined[rm] = 0
            refined = _select_component_centered(refined)

            # Re-grow edges after carving, but stay conservative.
            refined = _recover_trimmed_edges(
                rot_bgr,
                refined,
                sig,
                roi01=roi,
                cfg=cfg,
                runtime=runtime,
                kd_scale=1.35,
                ring_scale=1.15,
                max_area_mul=1.22,
            )

    # Shape-only carve for thumb-index webbing (works even when colors/shadows are ambiguous).
    if bool(cfg.webbing_convexity_enabled):
        # Border-reachable background evidence in the upright frame.
        reach_bg = _reachable_background_mask(sig, strictness=float(cfg.shadow_strictness), cfg=cfg, roi01=roi)
        reach_edge = _edge_reachable_map(sig, strength=float(cfg.reach_carve_strength))
        reach_bg = ((reach_bg > 0) | (reach_edge > 0)).astype(np.uint8)
        carve_stats: dict[str, float | str] = {}
        refined2 = _convexity_webbing_carve(refined, roi, (x, y, ww, hh), reach_bg, cfg, stats=carve_stats)
        if runtime is not None and carve_stats:
            runtime["carve_stats"] = carve_stats
        if refined2.sum() > 0:
            refined = refined2

    # Webbing/shadow pocket restore:
    # If the boundary band in finger ROI contains too many shadow/bg-like pixels, shrink and reconstruct
    # inside a constrained allow-set to avoid filling those pockets.
    if bool(cfg.webbing_restore_enabled):
        refined = (refined > 0).astype(np.uint8)
        k_restore = max(5, int(round(float(cfg.webbing_erode_k_frac) * float(min(refined.shape[:2])))))
        k_restore = int(min(19, max(5, k_restore | 1)))
        core = cv2.erode(refined, _k(k_restore), iterations=1)
        boundary = (refined & (1 - core)).astype(np.uint8)
        sel = (roi > 0) & (boundary > 0)
        denom = float(np.sum(sel)) + 1e-6
        bg_like = (sig.d_bg_u8.astype(np.float32) <= float(bg_like_thr)).astype(np.uint8)
        dt = cv2.distanceTransform((refined * 255).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        thin = (dt <= float(max(7.0, float(cfg.thin_dt_frac) * float(min(hh2, ww2))))).astype(np.uint8)
        bad = ((sel.astype(np.uint8) & thin) & ((bg_like > 0) | (shadow_any > 0))).astype(np.uint8)
        bad_ratio = float(bad.sum()) / float(denom)
        if bad_ratio >= float(cfg.webbing_trigger_ratio):
            forbid = (roi & boundary & thin & ((bg_like > 0) | (shadow_any > 0))).astype(np.uint8)
            allow = (refined & (1 - forbid)).astype(np.uint8)
            recon = _geodesic_reconstruct(core, allow, max_iters=int(cfg.webbing_max_iters))
            recon = _select_component_centered(recon)
            refined = recon

    # Final gap/shadow carve based on edge-reachable background.
    refined = _edge_reachable_bg_carve(refined, sig, roi01=roi, cfg=cfg)

    if int(tip_keep.sum()) > 0:
        refined = ((refined > 0) | (tip_keep > 0)).astype(np.uint8)
        refined = _select_component_centered(refined)

    inv = cv2.invertAffineTransform(mat)
    back = cv2.warpAffine(refined, inv, (m0.shape[1], m0.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)
    back = _select_component_centered((back > 0).astype(np.uint8))
    return back.astype(np.uint8)


def segment_glove(
    bgr: np.ndarray,
    cfg: SegmentationConfig | None = None,
    debug_out: dict[str, object] | None = None,
) -> SegmentationResult:
    """
    Produce a glove mask robust to moderate lighting/background changes.

    Returns:
    - glove_mask: raw mask (may contain holes)
    - glove_mask_filled: same mask with internal holes filled (useful for shape features)
    """
    cfg = SegmentationConfig() if cfg is None else cfg
    runtime: dict[str, object] = {} if debug_out is None else debug_out
    runtime.clear()
    runtime["prune_stats"] = []
    runtime["carve_stats"] = {}
    h, w = bgr.shape[:2]
    sig = _compute_signals(bgr)
    _, sure_bg = _bg_distance_candidate(sig)
    _candidates, _silhouette_ref, scored_base, candidate_validity = _prepare_candidates(bgr, sig, cfg)
    runtime["candidate_validity"] = candidate_validity
    runtime["candidate_area_frac"] = {k: float(v.get("area_frac", 0.0)) for k, v in candidate_validity.items()}
    runtime["candidate_iou_silhouette"] = {k: float(v.get("iou_silhouette", 0.0)) for k, v in candidate_validity.items()}

    # Always include a centered rectangle prior as last resort.
    rect = np.zeros((h, w), dtype=np.uint8)
    margin = int(round(min(h, w) * 0.08))
    rect[margin:-margin, margin:-margin] = 1

    best_method, best_init, best_score, best_metrics = scored_base[0]

    def _best_aux(exclude_method: str) -> np.ndarray:
        for m, mk, sc, _mt in scored_base:
            if m != exclude_method and float(sc) > -1e8:
                return mk
        for m, mk, _sc, _mt in scored_base:
            if m != exclude_method:
                return mk
        return best_init

    aux_init = _best_aux(best_method)
    if cfg.force_candidate in {m for (m, _mk, _sc, _mt) in scored_base}:
        for m, mk, sc, mt in scored_base:
            if m == cfg.force_candidate:
                best_method, best_init, best_score, best_metrics = m, mk, sc, mt
                aux_init = _best_aux(best_method)
                break

    # Use centered rectangle only as last resort; don't let it outscore real candidates.
    if float(best_score) < -1e8:
        best_method = "rect"
        best_init = rect
        aux_init = _best_aux(best_method) if len(scored_base) > 1 else rect

    # Adaptive refinement: prefer GrabCut when we have enough evidence; else use watershed.
    bg_sep = float(best_metrics.get("bg_sep_u8", 0.0)) if isinstance(best_metrics, dict) else 0.0
    edge_align = float(best_metrics.get("edge_align", 0.0)) if isinstance(best_metrics, dict) else 0.0
    use_grabcut = bool((bg_sep >= 12.0) or (edge_align >= 0.06))
    if cfg.force_refine == "grabcut":
        use_grabcut = True
    elif cfg.force_refine == "watershed":
        use_grabcut = False

    def _want_trimap(bg_sep_u8: float, edge_align_v: float) -> bool:
        if not bool(getattr(cfg, "grabcut_trimap_enabled", False)):
            return False
        if not bool(getattr(cfg, "grabcut_trimap_only_if_low_confidence", True)):
            return True
        return bool(
            (float(bg_sep_u8) < float(getattr(cfg, "grabcut_trimap_low_conf_bg_sep", 22.0)))
            and (float(edge_align_v) < float(getattr(cfg, "grabcut_trimap_low_conf_edge_align", 0.10)))
        )

    # Shadow suppression: treat shadow-like pixels around the seed as sure background
    # to prevent GrabCut/Watershed expansion into soft shadows.
    shadow01 = _shadow_like_mask(sig, strictness=float(cfg.shadow_strictness))
    seed_union = ((best_init > 0) | (aux_init > 0)).astype(np.uint8)
    seed_ring = (cv2.dilate(seed_union, _k(max(9, int(round(min(h, w) * 0.025)))), iterations=1) > 0).astype(np.uint8)
    seed_ring = (seed_ring & (1 - seed_union)).astype(np.uint8)
    sure_bg2 = (sure_bg | (shadow01 & seed_ring)).astype(np.uint8)

    # Global tip keep-mask derived from the *selected candidate*.
    # This is applied late (after aggressive carving stages) to prevent fingertip loss
    # cascading through later refinements.
    tip_keep_global = np.zeros((h, w), dtype=np.uint8)
    if bool(getattr(cfg, "tip_protect_enabled", True)) and best_method != "rect":
        tip_keep_global = _tip_keep_mask_upright(
            bgr,
            (best_init > 0).astype(np.uint8),
            frac=float(getattr(cfg, "tip_protect_frac", 0.18)),
            dilate_frac=float(getattr(cfg, "tip_protect_dilate_frac", 0.006)),
        )
        tip_keep_global = (tip_keep_global & (best_init > 0).astype(np.uint8)).astype(np.uint8)
        if runtime is not None:
            runtime["tip_keep_global_px"] = float(tip_keep_global.sum())

    refined01 = None
    used_refine = "grabcut" if use_grabcut else "watershed"
    if use_grabcut:
        use_trimap = _want_trimap(bg_sep, edge_align)
        runtime["grabcut_trimap_use"] = bool(use_trimap)
        refined01 = _grabcut_refine(
            bgr,
            best_init,
            aux_mask01=aux_init,
            sure_bg_mask01=sure_bg2,
            cfg=cfg,
            runtime=runtime,
            sig=sig,
            use_trimap=bool(use_trimap),
        )
    else:
        # Add shadow-like ring pixels as background markers by forcing them out post-hoc.
        refined01 = _watershed_refine(bgr, best_init, sig, cfg=cfg, runtime=runtime)
        refined01 = (refined01 & (1 - (shadow01 & seed_ring))).astype(np.uint8)

    # Final sanity: if the top choice is unreasonable, fall back to the next candidate.
    def _is_reasonable(mask01: np.ndarray) -> bool:
        if mask01.sum() < int(0.03 * h * w):
            return False
        if mask01.sum() > int(0.90 * h * w):
            return False
        if _border_touch_ratio(mask01) > 0.24:
            return False
        x, y, ww, hh = cv2.boundingRect(mask01.astype(np.uint8))
        extent = float(mask01.sum()) / float(ww * hh + 1e-6)
        return extent >= 0.12

    if not _is_reasonable(refined01):
        for m2, mk2, sc2, metrics2 in scored_base[1:]:
            if float(sc2) < -1e8:
                continue
            bg_sep2 = float(metrics2.get("bg_sep_u8", 0.0))
            edge_align2 = float(metrics2.get("edge_align", 0.0))
            use_gc2 = bool((bg_sep2 >= 12.0) or (edge_align2 >= 0.06))
            if cfg.force_refine == "grabcut":
                use_gc2 = True
            elif cfg.force_refine == "watershed":
                use_gc2 = False
            cand2 = (
                _grabcut_refine(
                    bgr,
                    mk2,
                    aux_mask01=best_init,
                    sure_bg_mask01=sure_bg2,
                    cfg=cfg,
                    runtime=runtime,
                    sig=sig,
                    use_trimap=bool(_want_trimap(bg_sep2, edge_align2)),
                )
                if use_gc2
                else _watershed_refine(bgr, mk2, sig, cfg=cfg, runtime=runtime)
            )
            if _is_reasonable(cand2):
                best_method = m2
                refined01 = cand2
                used_refine = "grabcut" if use_gc2 else "watershed"
                break

    # Recover slightly trimmed edges (especially for fabric/latex) in a controlled manner.
    if cfg.edge_recover_enabled:
        refined01 = _recover_trimmed_edges(
            bgr,
            refined01,
            sig,
            cfg=cfg,
            runtime=runtime,
            kd_scale=float(cfg.edge_recover_kd_scale),
            ring_scale=float(cfg.edge_recover_ring_scale),
            max_area_mul=float(cfg.edge_recover_max_area_mul),
        )

    # Optional glove-aware pass: rotate upright, recover fingertip edges, rotate back.
    # Keep only if it objectively improves the mask score under the same signals.
    used_upright = False
    sc0, _m0 = _score_mask(bgr, refined01, sig, cfg=cfg)
    upright = _upright_finger_refine(bgr, refined01, cfg, runtime=runtime) if cfg.upright_enabled else None
    if upright is not None and upright.sum() > 0 and cfg.upright_enabled:
        sc1, _m1 = _score_mask(bgr, upright, sig, cfg=cfg)
        # Add a glove-aware penalty: background-like pixels inside the finger ROI indicate filled valleys/webbing.
        rot_bgr0, rot_m0, _ = _upright_normalize(bgr, refined01)
        sig0 = _compute_signals(rot_bgr0)
        roi0, _bbox0 = _finger_roi_from_mask(rot_m0, frac=float(cfg.upright_roi_frac))
        bg_like0 = _bg_like_inside_ratio(rot_m0, sig0, roi0)

        rot_bgr1, rot_m1, _ = _upright_normalize(bgr, upright)
        sig1 = _compute_signals(rot_bgr1)
        roi1, _bbox1 = _finger_roi_from_mask(rot_m1, frac=float(cfg.upright_roi_frac))
        bg_like1 = _bg_like_inside_ratio(rot_m1, sig1, roi1)

        # Prefer the upright variant if it improves global score OR reduces bg-like finger pixels meaningfully,
        # while maintaining low border touch.
        better_global = float(sc1) > float(sc0) + float(cfg.upright_min_improve_score)
        better_gap = float(bg_like1) + float(cfg.upright_min_improve_bg_like) < float(bg_like0)
        safe = _border_touch_ratio(upright) <= max(0.12, _border_touch_ratio(refined01) + 0.03)
        tip_retain_ok = True
        tip_retain = 1.0
        if bool(getattr(cfg, "tip_protect_enabled", True)):
            # Compute retention in the upright-normalized frame to avoid ambiguity
            # about which end is the fingertips in the original orientation.
            xT, yT, wwT, hhT = cv2.boundingRect(rot_m0.astype(np.uint8))
            if wwT > 0 and hhT > 0:
                f = float(max(0.06, min(0.35, float(getattr(cfg, "tip_protect_frac", 0.18)))))
                band_h = int(max(6, round(f * float(hhT))))
                y_end_top = int(min(rot_m0.shape[0], max(0, yT + band_h)))
                y_start_bot = int(max(0, min(rot_m0.shape[0], (yT + hhT) - band_h)))
                tip_u = np.zeros_like(rot_m0, dtype=np.uint8)
                tip_u[yT:y_end_top, xT : xT + wwT] = 1
                tip_u[y_start_bot : yT + hhT, xT : xT + wwT] = 1
                tip_u = (tip_u & (rot_m0 > 0).astype(np.uint8)).astype(np.uint8)
                denom = float(tip_u.sum())
                if denom >= 120.0:
                    tip_retain = float(((rot_m1 > 0) & (tip_u > 0)).sum()) / float(denom + 1e-6)
                    tip_retain_ok = bool(tip_retain >= float(getattr(cfg, "upright_min_tip_retain", 0.92)))
        if runtime is not None:
            runtime["upright_tip_retain"] = float(tip_retain)
            runtime["upright_tip_retain_ok"] = bool(tip_retain_ok)

        if safe and tip_retain_ok and (better_global or better_gap):
            refined01 = upright
            used_upright = True

    # Global post-cleanups in original orientation:
    # - peel border-reachable background halo leaks,
    # - then restore tiny clipped fingertips (primarily leather).
    refined01 = _global_boundary_halo_peel(refined01, sig, cfg, runtime=runtime)
    refined01 = _tip_restore(refined01, sig, cfg, runtime=runtime)

    # ── Edge-based finger refinement ──────────────────────────────────
    # Uses Canny edges as barriers to carve inter-finger gaps (fabric) and
    # recover clipped fingertips (leather/latex).
    if bool(cfg.edge_finger_separation_enabled):
        refined01 = _edge_based_finger_refine(bgr, refined01, sig, cfg, runtime=runtime)

    if int(tip_keep_global.sum()) > 0:
        refined01 = ((refined01 > 0) | (tip_keep_global > 0)).astype(np.uint8)
        refined01 = _select_component_centered(refined01)

    # Seal small gaps so interior holes (true defects) remain enclosed and detectable.
    refined01 = cv2.morphologyEx(refined01, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    filled01 = _fill_holes(refined01)

    return SegmentationResult(
        glove_mask=(refined01 * 255).astype(np.uint8),
        glove_mask_filled=(filled01 * 255).astype(np.uint8),
        method=f"{best_method}+{used_refine}" + ("+upright_fingers" if used_upright else ""),
    )


def segment_glove_debug(bgr: np.ndarray, cfg: SegmentationConfig | None = None) -> tuple[SegmentationResult, SegmentationDebug]:
    """
    Debug version of segmentation: returns the segmentation result plus signals/candidates.
    """
    cfg = SegmentationConfig() if cfg is None else cfg
    sig = _compute_signals(bgr)
    candidates, silhouette_ref, scored_base, candidate_validity = _prepare_candidates(bgr, sig, cfg)
    candidates = {**candidates, "silhouette_ref": silhouette_ref}
    scored = []
    for m, _mk, sc, metrics in scored_base:
        valid = candidate_validity.get(m, {})
        scored.append(
            {
                "method": m,
                "score": float(sc),
                "candidate_valid": bool(valid.get("valid", True)),
                "candidate_area_frac": float(valid.get("area_frac", 0.0)),
                "candidate_iou_silhouette": float(valid.get("iou_silhouette", 0.0)),
                **(metrics if isinstance(metrics, dict) else {}),
            }
        )
    scored.sort(key=lambda d: float(d.get("score", -1e9)), reverse=True)

    runtime: dict[str, object] = {}
    res = segment_glove(bgr, cfg=cfg, debug_out=runtime)
    prune_summary: dict[str, float | bool | str] = {"applied": False, "reverted": False, "reason": "none"}
    prune_rows = runtime.get("prune_stats")
    if isinstance(prune_rows, list) and prune_rows:
        last = prune_rows[-1]
        if isinstance(last, dict):
            prune_summary = {k: v for k, v in last.items() if isinstance(v, (float, int, bool, str))}
    carve_summary: dict[str, float | str] = {"mode_used": "none", "evidence_removed_px": 0.0, "geometry_removed_px": 0.0}
    carve_row = runtime.get("carve_stats")
    if isinstance(carve_row, dict):
        carve_summary = {k: v for k, v in carve_row.items() if isinstance(v, (float, int, str))}
    halo_row = runtime.get("halo_peel")
    if isinstance(halo_row, dict):
        for k, v in halo_row.items():
            if isinstance(v, (float, int, str)):
                carve_summary[f"halo_{k}"] = v
    tip_row = runtime.get("tip_restore")
    if isinstance(tip_row, dict):
        for k, v in tip_row.items():
            if isinstance(v, (float, int, str)):
                carve_summary[f"tip_{k}"] = v

    trimap_use = bool(runtime.get("grabcut_trimap_use", False))
    trimap_stats: dict[str, float | int | bool | str] = {}
    trow = runtime.get("grabcut_trimap")
    if isinstance(trow, dict):
        trimap_stats = {k: v for k, v in trow.items() if isinstance(v, (float, int, bool, str))}

    dbg = SegmentationDebug(
        bg_is_white=bool(sig.bg_is_white),
        bg_is_uniform=bool(sig.bg_is_uniform),
        bg_lab_median=(float(sig.bg_med[0]), float(sig.bg_med[1]), float(sig.bg_med[2])),
        bg_lab_mad=(float(sig.bg_mad[0]), float(sig.bg_mad[1]), float(sig.bg_mad[2])),
        d_bg_u8=sig.d_bg_u8.copy(),
        grad_u8=sig.grad_u8.copy(),
        edges_u8=sig.edges_u8.copy(),
        candidates={k: v.copy() for k, v in candidates.items()},
        scored=scored,
        candidate_validity={k: dict(v) for k, v in candidate_validity.items()},
        carve_stats=carve_summary,
        prune_stats=prune_summary,
        grabcut_trimap_use=bool(trimap_use),
        grabcut_trimap=dict(trimap_stats),
        chosen_method=str(res.method),
    )
    return res, dbg
