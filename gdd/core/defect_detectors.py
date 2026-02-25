from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .anomaly import AnomalyMaps, build_anomaly_maps, candidate_blobs
from .defect_profiles import DefectTypeProfile, get_defect_profile
from .types import BoundingBox, Defect


@dataclass(frozen=True)
class DefectDetectionContext:
    bgr: np.ndarray
    glove_mask: np.ndarray
    glove_mask_filled: np.ndarray
    anomaly: AnomalyMaps
    glove_type: str
    profile: DefectTypeProfile
    specular_mask: np.ndarray
    quality: dict[str, float]


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _zero_anomaly_like(mask01: np.ndarray) -> AnomalyMaps:
    z = np.zeros(mask01.shape[:2], dtype=np.float32)
    return AnomalyMaps(color=z, texture=z, edges=z, combined=z)


def _specular_mask01(bgr: np.ndarray, glove_mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h = hsv[:, :, 0] / 179.0
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0
    mask01 = (glove_mask > 0).astype(np.uint8)
    # Saturation/value gating removes bright table reflections and narrow glare streaks.
    spec = (v > 0.88) & (s < 0.22) & (h >= 0.0)
    out = (spec & (mask01 > 0)).astype(np.uint8)
    if out.sum() == 0:
        return out
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return out


def _mask_quality(glove_mask: np.ndarray, bgr: np.ndarray) -> dict[str, float]:
    mask01 = (glove_mask > 0).astype(np.uint8)
    h, w = mask01.shape[:2]
    area = float(mask01.sum())
    area_frac = area / float((h * w) + 1e-6)
    if area <= 0:
        return {"area_frac": 0.0, "extent": 0.0, "border_touch": 1.0, "edge_align": 0.0, "ok_surface": 0.0, "ok_finger": 0.0}
    x, y, ww, hh = cv2.boundingRect(mask01)
    extent = float(area) / float((ww * hh) + 1e-6) if ww > 0 and hh > 0 else 0.0

    band = max(5, int(round(min(h, w) * 0.04)))
    border = np.zeros_like(mask01, dtype=np.uint8)
    border[:band, :] = 1
    border[-band:, :] = 1
    border[:, :band] = 1
    border[:, -band:] = 1
    border_touch = float((mask01 & border).sum()) / float(area + 1e-6)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    k = 5
    dil = cv2.dilate(mask01, np.ones((k, k), np.uint8), iterations=1)
    ero = cv2.erode(mask01, np.ones((k, k), np.uint8), iterations=1)
    ring = (dil > 0).astype(np.uint8) - (ero > 0).astype(np.uint8)
    edge_align = float(((ring > 0) & (edges > 0)).sum()) / float(ring.sum() + 1e-6)

    ok_surface = float(area_frac >= 0.05 and area_frac <= 0.85 and extent >= 0.25 and border_touch <= 0.12)
    ok_finger = float(area_frac >= 0.05 and area_frac <= 0.85 and edge_align >= 0.02)
    return {
        "area_frac": float(area_frac),
        "extent": float(extent),
        "border_touch": float(border_touch),
        "edge_align": float(edge_align),
        "ok_surface": float(ok_surface),
        "ok_finger": float(ok_finger),
    }


FOCUS_LABELS = {"missing_finger", "extra_fingers", "hole", "discoloration", "damaged_by_fold"}

LABEL_GROUP_HOLES = {"hole", "tear"}
LABEL_GROUP_SURFACE = {"stain_dirty", "spotting", "plastic_contamination"}  # discoloration handled separately
LABEL_GROUP_WRINKLE = {"wrinkles_dent", "damaged_by_fold"}
LABEL_GROUP_CUFF = {"improper_roll", "incomplete_beading"}
LABEL_GROUP_FINGERS = {"missing_finger", "extra_fingers"}

# Labels whose detector(s) require the anomaly maps.
LABELS_NEED_ANOMALY = set(LABEL_GROUP_SURFACE) | set(LABEL_GROUP_WRINKLE)


def _rotate_mask_to_upright(mask01: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Rotate the binary mask so the main axis of the glove is roughly vertical.
    Returns (rotated_mask01, angle_degrees_applied).
    """
    pts = np.column_stack(np.where(mask01 > 0))  # (y,x)
    if pts.shape[0] < 200:
        return mask01, 0.0, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    pts_xy = pts[:, ::-1].astype(np.float32)  # (x,y)
    mean = pts_xy.mean(axis=0, keepdims=True)
    centered = pts_xy - mean
    cov = (centered.T @ centered) / float(max(1, centered.shape[0] - 1))
    vals, vecs = np.linalg.eigh(cov)
    pc1 = vecs[:, int(np.argmax(vals))]  # (vx,vy)

    angle = float(np.degrees(np.arctan2(pc1[1], pc1[0])))  # relative to +x
    # We want the main axis to align with +y (90 degrees).
    rot = 90.0 - angle

    h, w = mask01.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rot, 1.0)
    out = cv2.warpAffine(mask01.astype(np.uint8), m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return (out > 0).astype(np.uint8), float(rot), m.astype(np.float32)


def _profile_peaks(mask01: np.ndarray) -> tuple[list[int], np.ndarray, int, int]:
    """
    Count finger-like peaks by projecting the top portion of the mask onto the x-axis.
    This is tolerant to lighting changes since it uses only silhouette geometry.
    """
    if mask01.sum() < 300:
        return [], np.zeros((mask01.shape[1],), dtype=np.float32), 0, 0

    ys = np.where(mask01 > 0)[0]
    if ys.size == 0:
        return [], np.zeros((mask01.shape[1],), dtype=np.float32), 0, 0
    y0, y1 = int(ys.min()), int(ys.max())
    h = y1 - y0 + 1
    top = mask01[y0 : y0 + int(round(0.50 * h)), :]
    if top.sum() < 200:
        return [], np.zeros((mask01.shape[1],), dtype=np.float32), int(y0), int(h)

    prof = top.sum(axis=0).astype(np.float32)
    if prof.max() <= 1:
        return [], np.zeros((mask01.shape[1],), dtype=np.float32), int(y0), int(h)
    prof /= (prof.max() + 1e-6)

    win = 11
    kernel = np.ones((win,), np.float32) / float(win)
    prof_s = np.convolve(prof, kernel, mode="same")

    thresh = max(0.25, float(np.percentile(prof_s, 75)))
    min_sep = max(14, int(round(mask01.shape[1] * 0.05)))

    peaks: list[int] = []
    for i in range(2, len(prof_s) - 2):
        if prof_s[i] < thresh:
            continue
        if prof_s[i] >= prof_s[i - 1] and prof_s[i] >= prof_s[i + 1] and prof_s[i] >= prof_s[i - 2] and prof_s[i] >= prof_s[i + 2]:
            if not peaks or (i - peaks[-1]) > min_sep:
                peaks.append(i)
    return peaks, prof_s.astype(np.float32), int(y0), int(h)


def _profile_peaks_param(
    mask01: np.ndarray,
    *,
    top_frac: float,
    win: int,
    percentile: float,
    thresh_floor: float,
    min_sep_frac: float,
) -> tuple[list[int], np.ndarray, int, int]:
    """
    Variant of `_profile_peaks` with tunable smoothing/thresholding.
    Used for fabric-specific finger counting where peaks can merge under heavy mask blur.
    """
    if mask01.sum() < 300:
        return [], np.zeros((mask01.shape[1],), dtype=np.float32), 0, 0

    ys = np.where(mask01 > 0)[0]
    if ys.size == 0:
        return [], np.zeros((mask01.shape[1],), dtype=np.float32), 0, 0
    y0, y1 = int(ys.min()), int(ys.max())
    h = y1 - y0 + 1
    top = mask01[y0 : y0 + int(round(float(top_frac) * h)), :]
    if top.sum() < 200:
        return [], np.zeros((mask01.shape[1],), dtype=np.float32), int(y0), int(h)

    prof = top.sum(axis=0).astype(np.float32)
    if prof.max() <= 1:
        return [], np.zeros((mask01.shape[1],), dtype=np.float32), int(y0), int(h)
    prof /= (prof.max() + 1e-6)

    win = int(max(3, win))
    if win % 2 == 0:
        win += 1
    kernel = np.ones((win,), np.float32) / float(win)
    prof_s = np.convolve(prof, kernel, mode="same")

    percentile = float(np.clip(percentile, 5.0, 95.0))
    thresh = max(float(thresh_floor), float(np.percentile(prof_s, percentile)))
    min_sep = max(10, int(round(mask01.shape[1] * float(min_sep_frac))))

    peaks: list[int] = []
    for i in range(2, len(prof_s) - 2):
        if prof_s[i] < thresh:
            continue
        if prof_s[i] >= prof_s[i - 1] and prof_s[i] >= prof_s[i + 1] and prof_s[i] >= prof_s[i - 2] and prof_s[i] >= prof_s[i + 2]:
            if not peaks or (i - peaks[-1]) > min_sep:
                peaks.append(i)
    return peaks, prof_s.astype(np.float32), int(y0), int(h)


def _count_profile_peaks(mask01: np.ndarray) -> int:
    peaks, _prof, _y0, _h = _profile_peaks(mask01)
    return int(len(peaks))


def _top_profile(mask01: np.ndarray) -> np.ndarray:
    """
    For each x, return the top-most y in the silhouette (or -1 if empty column).
    """
    h, w = mask01.shape[:2]
    prof = np.full((w,), -1, dtype=np.int32)
    for x in range(w):
        ys = np.where(mask01[:, x] > 0)[0]
        if ys.size:
            prof[x] = int(ys.min())
    return prof


def _longest_true_run(flags: np.ndarray) -> int:
    arr = np.asarray(flags).astype(np.uint8).reshape(-1)
    best = 0
    run = 0
    for v in arr.tolist():
        if int(v) > 0:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return int(best)


def _bbox_from_rotated_box(bbox_rot: BoundingBox, inv_m: np.ndarray, w: int, h: int) -> BoundingBox | None:
    x1, y1, x2, y2 = bbox_rot.as_xyxy()
    corners = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], dtype=np.float32)
    pts = cv2.transform(corners.reshape(1, -1, 2), inv_m).reshape(-1, 2)
    min_x = int(np.floor(np.min(pts[:, 0])))
    min_y = int(np.floor(np.min(pts[:, 1])))
    max_x = int(np.ceil(np.max(pts[:, 0])))
    max_y = int(np.ceil(np.max(pts[:, 1])))
    min_x = max(0, min(w - 1, min_x))
    min_y = max(0, min(h - 1, min_y))
    max_x = max(0, min(w - 1, max_x))
    max_y = max(0, min(h - 1, max_y))
    bw = max(0, max_x - min_x)
    bh = max(0, max_y - min_y)
    if bw < 5 or bh < 5:
        return None
    return BoundingBox(x=min_x, y=min_y, w=bw, h=bh)


def _bbox_iou(a: BoundingBox | None, b: BoundingBox | None) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a.as_xyxy()
    bx1, by1, bx2, by2 = b.as_xyxy()
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0.0:
        return 0.0
    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    return float(inter / (area_a + area_b - inter + 1e-6))


def _count_fingers_convexity(mask01: np.ndarray) -> int:
    """
    Count fingers using convexity defects on the glove silhouette.
    Rough idea (as commonly suggested for hand/finger counting in OpenCV):
      valleys between fingers -> convexity defects -> finger count ~ defects + 1
    """
    m = (mask01 > 0).astype(np.uint8)
    if m.sum() < 400:
        return 0

    cnts, _ = cv2.findContours((m * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 400:
        return 0

    hull = cv2.convexHull(c, returnPoints=False)
    if hull is None or len(hull) < 3:
        return 0
    try:
        defects = cv2.convexityDefects(c, hull)
    except cv2.error:
        c2 = cv2.approxPolyDP(c, epsilon=2.0, closed=True)
        if c2 is None or len(c2) < 4:
            return 0
        hull2 = cv2.convexHull(c2, returnPoints=False)
        if hull2 is None or len(hull2) < 3:
            return 0
        try:
            defects = cv2.convexityDefects(c2, hull2)
            c = c2
        except cv2.error:
            return 0
    if defects is None:
        return 0

    ys = c[:, 0, 1]
    y_min = int(np.min(ys))
    y_max = int(np.max(ys))
    h = max(1, y_max - y_min + 1)
    top_band_y = y_min + int(round(0.55 * h))

    good_valleys = 0
    for s, e, f, depth in defects.reshape(-1, 4):
        start = c[int(s)][0]
        end = c[int(e)][0]
        far = c[int(f)][0]
        depth_px = float(depth) / 256.0
        if depth_px < 10.0:
            continue
        # Focus on the upper part where finger gaps exist.
        if int(far[1]) > top_band_y:
            continue

        a = np.linalg.norm(end - start) + 1e-6
        b = np.linalg.norm(far - start) + 1e-6
        d = np.linalg.norm(end - far) + 1e-6
        # Angle at the far point.
        cosang = float((b * b + d * d - a * a) / (2.0 * b * d + 1e-6))
        cosang = max(-1.0, min(1.0, cosang))
        ang = float(np.degrees(np.arccos(cosang)))
        if ang < 18.0 or ang > 95.0:
            continue
        good_valleys += 1

    # Defects correspond to valleys between fingers; convert to count estimate.
    if good_valleys <= 0:
        return 0
    return int(min(8, max(0, good_valleys + 1)))


def _fingertips_polar_signature(
    mask01: np.ndarray,
    *,
    y0: int,
    gh: int,
    top_frac: float = 0.60,
    bins: int = 360,
    smooth_win: int = 9,
    peak_percentile: float = 78.0,
    peak_floor: float = 0.55,
    min_sep_frac: float = 0.07,
) -> tuple[list[int], dict]:
    """
    Count fingertip-like lobes using a polar contour signature (radius vs angle)
    in the upper part of the glove.

    This is intended as a *fabric-only rescue* for extra_fingers where adjacent
    fingers can merge under silhouette-based peak counters.
    """
    if gh <= 0 or mask01.sum() < 400:
        return [], {"mode": "polar", "reason": "empty"}

    m = (mask01 > 0).astype(np.uint8)
    cnts, _ = cv2.findContours((m * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return [], {"mode": "polar", "reason": "no_contour"}
    c = max(cnts, key=cv2.contourArea)
    if float(cv2.contourArea(c)) < 400.0:
        return [], {"mode": "polar", "reason": "small"}

    M = cv2.moments(c)
    if float(M.get("m00", 0.0)) <= 0.0:
        return [], {"mode": "polar", "reason": "bad_moments"}
    cx = float(M["m10"]) / float(M["m00"])
    cy = float(M["m01"]) / float(M["m00"])

    bins = int(max(60, bins))
    y_cut = int(y0 + int(round(float(top_frac) * float(gh))))
    y_cut = max(0, min(int(mask01.shape[0]) - 1, y_cut))

    pts = c.reshape(-1, 2).astype(np.float32)  # (x,y)
    pts = pts[pts[:, 1] <= float(y_cut)]
    if pts.shape[0] < 80:
        return [], {"mode": "polar", "reason": "insufficient_top_pts", "y_cut": int(y_cut)}

    dx = pts[:, 0] - float(cx)
    dy = pts[:, 1] - float(cy)
    ang = np.arctan2(dy, dx)  # [-pi, pi]
    idx = (((ang + np.pi) / (2.0 * np.pi)) * float(bins)).astype(np.int32)
    idx = np.clip(idx, 0, bins - 1)
    rad = np.sqrt(dx * dx + dy * dy).astype(np.float32)

    r = np.zeros((bins,), dtype=np.float32)
    np.maximum.at(r, idx, rad)
    nz = np.where(r > 0.0)[0]
    if nz.size < 10:
        return [], {"mode": "polar", "reason": "sparse_bins", "nz_bins": int(nz.size)}

    # Fill empty bins using nearest-neighbor (bins is small, so this is cheap and deterministic).
    for i in range(bins):
        if r[i] > 0.0:
            continue
        j = int(nz[np.argmin(np.abs(nz - i))])
        r[i] = float(r[j])

    r /= (float(r.max()) + 1e-6)
    smooth_win = int(max(3, smooth_win))
    if smooth_win % 2 == 0:
        smooth_win += 1
    kernel = np.ones((smooth_win,), np.float32) / float(smooth_win)
    rs = np.convolve(r, kernel, mode="same").astype(np.float32)

    peak_percentile = float(np.clip(peak_percentile, 5.0, 95.0))
    thr = max(float(peak_floor), float(np.percentile(rs, peak_percentile)))
    min_sep = max(12, int(round(float(bins) * float(min_sep_frac))))

    peaks: list[int] = []
    for i in range(2, bins - 2):
        if float(rs[i]) < thr:
            continue
        if rs[i] >= rs[i - 1] and rs[i] >= rs[i + 1] and rs[i] >= rs[i - 2] and rs[i] >= rs[i + 2]:
            if not peaks or (i - peaks[-1]) > min_sep:
                peaks.append(i)

    meta = {
        "mode": "polar",
        "bins": int(bins),
        "top_frac": float(top_frac),
        "smooth_win": int(smooth_win),
        "peak_percentile": float(peak_percentile),
        "thr": float(thr),
        "min_sep": int(min_sep),
        "y_cut": int(y_cut),
        "top_pts": int(pts.shape[0]),
        "nz_bins": int(nz.size),
    }
    return peaks, meta


def _estimate_finger_count(glove_mask_filled: np.ndarray) -> tuple[int, dict]:
    """
    Estimate how many fingers are present using two silhouette-only methods:
    - profile peak counting
    - convexity-defects counting
    """
    mask = (glove_mask_filled > 0).astype(np.uint8)
    rot_mask, rot, _m = _rotate_mask_to_upright(mask)
    p = _count_profile_peaks(rot_mask)
    h = _count_fingers_convexity(rot_mask)

    counts = [c for c in (p, h) if c > 0]
    if not counts:
        return 0, {"rot_deg": rot, "profile_peaks": p, "hull_count": h}
    # Use a robust summary (median-ish).
    counts.sort()
    est = counts[len(counts) // 2]
    return int(est), {"rot_deg": rot, "profile_peaks": p, "hull_count": h}


def _bbox_from_stats(stats_row: np.ndarray) -> BoundingBox:
    x, y, w, h, _ = [int(v) for v in stats_row.tolist()]
    return BoundingBox(x=x, y=y, w=w, h=h)


def _detect_latex_dark_puncture(
    bgr: np.ndarray,
    glove_mask_filled: np.ndarray,
) -> Defect | None:
    """
    Latex-only recall rescue for small, low-contrast punctures that do not appear as
    (filled_silhouette - raw_mask) voids and are not background-colored enough for
    the "background-like interior pixels" fallback in `_detect_holes`.

    This detector is intentionally conservative (high thresholds + strong interior gating)
    to avoid boundary/cuff artifacts and texture-driven false positives.
    """
    mf = (glove_mask_filled > 0).astype(np.uint8)
    if int(mf.sum()) < 800:
        return None

    dt = cv2.distanceTransform((mf * 255).astype(np.uint8), cv2.DIST_L2, 5)
    ys = np.where(mf > 0)[0]
    if ys.size:
        y0, y1 = int(ys.min()), int(ys.max())
        gh = max(1, y1 - y0 + 1)
        cuff_y = y1 - int(round(0.18 * gh))
    else:
        cuff_y = int(mf.shape[0] * 0.85)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]

    # Black-hat highlights small dark blobs relative to their neighborhood.
    bh = cv2.morphologyEx(
        L,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
    )

    interior = (mf > 0) & (dt >= 6.0)
    if int(interior.sum()) < 500:
        interior = (mf > 0) & (dt >= 4.0)
    interior = interior & (np.arange(mf.shape[0])[:, None] < int(cuff_y))
    if int(interior.sum()) < 300:
        return None

    vals = bh[interior]
    if vals.size < 300:
        return None
    thr = float(np.percentile(vals, 99.7))
    thr = float(np.clip(max(82.0, thr), 70.0, 140.0))

    cand = ((bh >= thr) & interior).astype(np.uint8)
    if int(cand.sum()) < 25:
        return None
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    glove_area = float(mf.sum()) + 1e-6
    num, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    best: Defect | None = None
    for i in range(1, num):
        x, y, w, h, area = [int(v) for v in stats[i].tolist()]
        if area < 45 or area > 260:
            continue
        aspect = float(max(w, h) / (min(w, h) + 1e-6))
        if aspect > 1.65:
            continue
        area_norm = float(area) / float(glove_area)
        if area_norm < 0.00015 or area_norm > 0.0016:
            continue

        comp = (labels == i).astype(np.uint8)
        if int(comp.sum()) < 25:
            continue
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        peri = float(cv2.arcLength(c, True)) + 1e-6
        circ = float((4.0 * np.pi * float(area)) / (peri * peri))
        if circ < 0.72:
            continue

        dt_vals = dt[comp > 0]
        if dt_vals.size <= 0:
            continue
        med_dt = float(np.median(dt_vals))
        if med_dt < 8.0:
            continue
        # Be conservative: most dark interior blobs on clean latex are shading/folds.
        # The FN we need to recover is a fingertip-adjacent puncture; constrain this rescue
        # to moderately interior-but-not-deep regions to avoid widespread false positives.
        if med_dt > 35.0:
            continue

        # Require a strong local darkening compared to the immediate neighborhood.
        dil = cv2.dilate(comp, np.ones((9, 9), np.uint8), iterations=1)
        ring = ((dil > 0).astype(np.uint8) - comp) & mf
        if int(ring.sum()) < 25:
            continue
        l_in = float(np.mean(L[comp > 0]))
        l_ring = float(np.mean(L[ring > 0]))
        ring_ldiff = float(l_ring - l_in)
        if ring_ldiff < 42.0:
            continue

        score = _clamp01(
            0.58
            + 0.22 * min(1.0, max(0.0, (circ - 0.72) / 0.28))
            + 0.16 * min(1.0, ring_ldiff / 85.0)
            + 0.10 * min(1.0, med_dt / 22.0)
        )
        d = Defect(
            label="hole",
            score=float(score),
            bbox=BoundingBox(x=int(x), y=int(y), w=int(w), h=int(h)),
            meta={
                "source": "latex_dark_puncture",
                "area_norm": float(area_norm),
                "circularity": float(circ),
                "aspect": float(aspect),
                "median_dt": float(med_dt),
                "ring_ldiff": float(ring_ldiff),
                "thr": float(thr),
            },
        )
        if best is None or float(d.score) > float(best.score):
            best = d
    return best


def _detect_holes(
    bgr: np.ndarray,
    glove_mask: np.ndarray,
    glove_mask_filled: np.ndarray,
    glove_type: str = "unknown",
) -> list[Defect]:
    """
    Detect holes/tears by finding regions that are inside the filled silhouette but missing in the raw mask.
    """
    m = (glove_mask > 0).astype(np.uint8)
    mf = (glove_mask_filled > 0).astype(np.uint8)
    glove_type_norm = str(glove_type or "unknown").strip().lower()
    glove_area = float(mf.sum()) + 1e-6
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    fallback_used = False
    dt_mf: np.ndarray | None = cv2.distanceTransform((mf * 255).astype(np.uint8), cv2.DIST_L2, 5)

    core = cv2.erode(mf, np.ones((9, 9), np.uint8), iterations=1)
    if core.sum() < 100:
        core = mf.copy()
    glove_med = np.median(lab[core > 0], axis=0) if core.sum() else np.array([0.0, 0.0, 0.0], dtype=np.float32)

    ring = cv2.dilate(mf, np.ones((25, 25), np.uint8), iterations=1) - mf
    border = np.zeros_like(mf, dtype=np.uint8)
    b = max(8, int(round(min(mf.shape[:2]) * 0.06)))
    border[:b, :] = 1
    border[-b:, :] = 1
    border[:, :b] = 1
    border[:, -b:] = 1
    bg_mask = ((ring > 0) | (border > 0)).astype(np.uint8)
    if bg_mask.sum() < 120:
        bg_mask = (mf == 0).astype(np.uint8)
    bg_med = np.median(lab[bg_mask > 0], axis=0) if bg_mask.sum() else glove_med.copy()

    holes = ((mf > 0) & (m == 0)).astype(np.uint8)
    if holes.sum() == 0 and mf.sum() > 0:
        # Fallback: some segmentations paint over interior voids (especially bright holes),
        # so explicitly search for background-like pixels inside the filled silhouette.
        fallback_used = True
        d_bg_px = np.linalg.norm(lab - bg_med.reshape(1, 1, 3), axis=2)
        d_gl_px = np.linalg.norm(lab - glove_med.reshape(1, 1, 3), axis=2)
        l_diff = np.abs(lab[:, :, 0] - float(glove_med[0]))
        # CLAHE (used in preprocess) can flip the apparent brightness of the hole/background region.
        # Use a symmetric gate so we keep both "brighter than glove" and "darker than glove" candidates.
        l_gate = (l_diff >= 10.0)

        # Adaptive background distance threshold:
        # estimate how "tight" the background color is far away from the glove boundary,
        # then accept interior candidates that are within a multiple of that typical spread.
        inv = (mf == 0).astype(np.uint8)
        dt_bg = cv2.distanceTransform((inv * 255).astype(np.uint8), cv2.DIST_L2, 5)
        bg_far = (inv > 0) & (dt_bg >= max(12.0, 0.02 * float(min(mf.shape[:2]))))
        if int(bg_far.sum()) >= 200:
            bg_d = d_bg_px[bg_far]
            # Use a mid-high percentile to avoid being dominated by CLAHE artifacts in flat backgrounds.
            # If we pick too high a percentile, we accept glove highlights as "background-like" and
            # create huge connected components.
            bg_p80 = float(np.percentile(bg_d, 80))
            thr_bg = float(np.clip(bg_p80 * 1.8, 18.0, 45.0))
        else:
            thr_bg = 26.0

        holes = (
            (mf > 0)
            & (dt_mf >= 4.0)
            & (d_bg_px <= thr_bg)
            & ((d_bg_px + 4.0) < d_gl_px)
            & l_gate
        ).astype(np.uint8)
    if holes.sum() == 0:
        return []
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    out: list[Defect] = []
    tear_candidates: list[Defect] = []
    # Cuff/opening suppression: ignore a very large "hole" near the bottom band
    # (often just the glove opening showing background).
    ys = np.where(mf > 0)[0]
    xs = np.where(mf > 0)[1]
    if ys.size:
        y0, y1 = int(ys.min()), int(ys.max())
        gh = max(1, y1 - y0 + 1)
        cuff_y = y1 - int(round(0.18 * gh))
    else:
        cuff_y = int(mf.shape[0] * 0.85)
    if xs.size:
        x0, x1 = int(xs.min()), int(xs.max())
        gw = max(1, x1 - x0 + 1)
    else:
        gw = mf.shape[1]

    for i in range(1, num):
        x, y, w, h, area = [int(v) for v in stats[i].tolist()]
        if area < 40:
            continue
        bbox = BoundingBox(x=x, y=y, w=w, h=h)
        aspect = (max(w, h) / (min(w, h) + 1e-6))
        area_norm = float(area) / glove_area

        cy = y + (h / 2.0)
        if cy >= float(cuff_y):
            # Likely cuff opening rather than a puncture/hole defect.
            if area_norm > 0.02:
                continue
            # Some segmentations leak a smaller-but-wide background region at the opening.
            if float(w) >= 0.22 * float(gw) and area_norm > 0.004:
                continue

        comp = (labels == i).astype(np.uint8)
        hole_med = np.median(lab[comp > 0], axis=0) if comp.sum() else glove_med
        d_bg = float(np.linalg.norm(hole_med - bg_med))
        d_glove = float(np.linalg.norm(hole_med - glove_med))
        l_abs = float(abs(float(glove_med[0]) - float(hole_med[0])))
        ring2 = cv2.dilate(comp, np.ones((9, 9), np.uint8), iterations=1) - comp
        ring2 = (ring2 > 0).astype(np.uint8)
        ring2 = (ring2 & mf).astype(np.uint8)
        ring_d_bg = 0.0
        de_ring = 0.0
        ring_l_abs = 0.0
        if int(ring2.sum()) >= 30:
            ring_med = np.median(lab[ring2 > 0], axis=0)
            ring_d_bg = float(np.linalg.norm(ring_med - bg_med))
            de_ring = float(np.linalg.norm(hole_med - ring_med))
            ring_l_abs = float(abs(float(hole_med[0]) - float(ring_med[0])))

        # Reject color anomalies that are not background-like (common false "hole" on stains/discoloration).
        if not (d_bg + 8.0 < d_glove and l_abs > 8.0):
            if area_norm > 0.0008:
                continue

        if dt_mf is not None and int(comp.sum()) > 0:
            mean_dt = float(dt_mf[comp > 0].mean())
        else:
            mean_dt = 0.0

        if fallback_used:
            # In fallback mode we are essentially "reverse segmenting" interior background pixels.
            # Be strict to avoid picking up leather highlights/seams.
            # 1) Surrounding ring should be non-background (so we don't pick a random patch of table),
            # and the candidate should be meaningfully different from that ring.
            if int(ring2.sum()) >= 40:
                # Ring should not look like background, and the component should stand out from ring.
                if not (ring_d_bg >= 14.0 and de_ring >= 16.0 and (d_bg + 6.0) < ring_d_bg):
                    continue

            # 2) Very small blobs are almost always specular highlights.
            if area_norm < 0.00035:
                continue
            # 3) Reverse-segmented candidates too close to boundary are usually edge leaks.
            if mean_dt < 2.1:
                continue
        else:
            # Direct segmentation path: edge leaks are common around mask boundaries; ignore them.
            if mean_dt < 1.9:
                continue

        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            peri = float(cv2.arcLength(c, True)) + 1e-6
            circ = float((4.0 * np.pi * float(area)) / (peri * peri))
        else:
            circ = 0.0

        # Tear-like cue: elongated/irregular void. Keep separately so hole logic stays puncture-oriented.
        if (
            (float(aspect) >= 1.9 or float(circ) <= 0.32)
            and area_norm >= 0.00025
            and area_norm <= 0.010
            and mean_dt >= 1.8
            and (d_bg + 5.0 < d_glove)
        ):
            tear_score = _clamp01(0.54 + 0.24 * min(1.0, max(0.0, float(aspect) - 1.4)) + 0.20 * min(1.0, mean_dt / 6.0))
            tear_candidates.append(
                Defect(
                    label="tear",
                    score=tear_score,
                    bbox=bbox,
                    meta={
                        "area": area,
                        "aspect": float(aspect),
                        "circularity": float(circ),
                        "area_norm": area_norm,
                        "d_bg": d_bg,
                        "d_glove": d_glove,
                        "mean_dt": mean_dt,
                        "source": "hole_void_shape",
                    },
                )
            )

        if fallback_used:
            # Specular highlights and small seam gaps can look "background-like" on leather/fabric,
            # but they rarely form a compact closed shape. Enforce moderate circularity for holes.
            if aspect < 2.2 and float(circ) < 0.50:
                continue

        if glove_type_norm == "latex":
            latex_ring_ok = (d_bg >= 30.0) and (ring_d_bg >= 45.0) and (float(circ) >= 0.60)
            latex_deep_ok = (mean_dt >= 60.0) and (float(circ) >= 0.60)
            if not (latex_ring_ok or latex_deep_ok):
                continue
        else:
            # Recall-oriented: allow slightly less circular interior voids (common in AI tears/holes).
            if float(circ) < 0.50:
                continue
            if not (mean_dt >= 14.0 or area_norm >= 0.0007):
                continue

        # Confidence heuristic: bigger voids + high circularity are strong evidence.
        size_score = min(1.0, area_norm / 0.008)  # 0.8% of glove area → max confidence
        # Keep this detector focused on puncture-like holes.
        if area_norm > 0.0060:
            continue
        if area_norm < 0.00020 and mean_dt < 12.0:
            continue
        if aspect >= 2.2:
            continue
        if circ < 0.36:
            continue
        oversize_pen = max(0.0, (area_norm - 0.0030) / 0.0030)
        score = 0.50 + 0.42 * (0.52 * size_score + 0.33 * min(1.0, max(0.0, circ)) + 0.15 * min(1.0, mean_dt / 8.0)) - 0.18 * min(1.0, oversize_pen)
        out.append(
            Defect(
                label="hole",
                score=_clamp01(score),
                bbox=bbox,
                meta={
                    "area": area,
                    "aspect": float(aspect),
                    "circularity": circ,
                    "area_norm": area_norm,
                    "d_bg": d_bg,
                    "d_glove": d_glove,
                    "mean_dt": mean_dt,
                    "de_ring": de_ring,
                    "ring_d_bg": ring_d_bg,
                    "ring_l_abs": ring_l_abs,
                },
            )
        )
    out.extend(tear_candidates)

    # Recall rescue (latex/leather): some punctures are rendered as small edge-loops with almost no
    # background-like pixels inside the filled mask (fallback misses) and no silhouette void.
    # Detect small circular edge-loops well inside the glove and promote them to low-confidence holes.
    if glove_type_norm in {"latex", "leather"} and not any(d.label == "hole" for d in out):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 60, 160)
        cnts, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        best_loop: Defect | None = None
        for c in cnts:
            a = float(cv2.contourArea(c))
            if a < 50.0 or a > 900.0:
                continue
            x, y, w, h = cv2.boundingRect(c)
            max_wh = 32 if glove_type_norm == "latex" else 45
            if w < 6 or h < 6 or w > max_wh or h > max_wh:
                continue
            aspect = float(max(w, h) / (min(w, h) + 1e-6))
            if aspect > 1.55:
                continue
            peri = float(cv2.arcLength(c, True)) + 1e-6
            circ = float((4.0 * np.pi * a) / (peri * peri))
            if circ < 0.62:
                continue
            sub = mf[y : y + h, x : x + w]
            if sub.size <= 0:
                continue
            mask_frac = float((sub > 0).sum()) / float(sub.size + 1e-6)
            if mask_frac < 0.92:
                continue
            dt_sub = dt_mf[y : y + h, x : x + w]
            dt_vals = dt_sub[sub > 0] if dt_sub.size and int((sub > 0).sum()) else np.array([], dtype=np.float32)
            if dt_vals.size <= 0:
                continue
            med_dt = float(np.median(dt_vals))
            if med_dt < (10.0 if glove_type_norm == "latex" else 8.0):
                continue
            if glove_type_norm == "latex":
                # Reject loops where the interior chroma differs substantially from its neighborhood.
                # These are frequently seams/panel edges (and segmentation artifacts) rather than punctures.
                roi = np.zeros_like(mf, dtype=np.uint8)
                roi[y : y + h, x : x + w] = 1
                roi = (roi & mf).astype(np.uint8)
                if int(roi.sum()) <= 12:
                    continue
                dil = cv2.dilate(roi, np.ones((9, 9), np.uint8), iterations=1)
                ring = ((dil > 0).astype(np.uint8) - roi) & mf
                if int(ring.sum()) >= 20:
                    ab = lab[:, :, 1:3]
                    ab_roi = ab[roi > 0]
                    ab_ring = ab[ring > 0]
                    if ab_roi.size and ab_ring.size:
                        dab = float(np.linalg.norm(np.mean(ab_roi, axis=0) - np.mean(ab_ring, axis=0)))
                        if dab > 4.25:
                            continue
            area_norm = float(a) / float(glove_area + 1e-6)
            # Extremely tiny loops are usually pinholes in the AI mask or texture creases, not real punctures.
            if area_norm < (0.00040 if glove_type_norm == "latex" else 0.00055):
                continue
            score = _clamp01(0.50 + 0.30 * min(1.0, circ / 0.90) + 0.20 * min(1.0, med_dt / 22.0))
            d = Defect(
                label="hole",
                score=score,
                bbox=BoundingBox(x=int(x), y=int(y), w=int(w), h=int(h)),
                meta={
                    "source": f"{glove_type_norm}_edge_loop",
                    "area_norm": float(area_norm),
                    "circularity": float(circ),
                    "aspect": float(aspect),
                    "median_dt": float(med_dt),
                },
            )
            if best_loop is None or float(d.score) > float(best_loop.score):
                best_loop = d
        if best_loop is not None:
            out.append(best_loop)

    # Latex-only recall rescue: small interior punctures can be rendered as local dark blobs
    # without creating a segmentation void. Prefer this interior cue over boundary-notch logic.
    if glove_type_norm == "latex" and not any(d.label == "hole" for d in out):
        punct = _detect_latex_dark_puncture(bgr, glove_mask_filled)
        if punct is not None:
            out.append(punct)

    # If the segmentation gives us true interior voids (holes_direct path),
    # multiple disconnected voids are plausible for perforations. Keep them all.
    # If we're in fallback mode (segmenter painted over the hole), the candidate
    # set tends to include specular highlights; be conservative and keep only one
    # best hole/tear for the image.
    if fallback_used and out:
        holes_out = [d for d in out if d.label == "hole"]
        if not holes_out:
            tears_out = [d for d in out if d.label == "tear"]
            if tears_out:
                return [max(tears_out, key=lambda d: float(d.score))]
            return []
        # In fallback mode, hole-vs-tear distinction is unreliable; return the strongest hole-like candidate.
        return [max(holes_out, key=lambda d: float(d.score))]
    if not out:
        # Last-resort recall fallback: some "hole" samples are rendered as boundary rips with no interior void.
        # Promote the strongest tear-notch cue into a low-confidence hole candidate.
        notch = _detect_tear_notches(glove_mask_filled)
        # Keep this ONLY for latex where small fingertip punctures can be partially clipped by segmentation,
        # and suppress it for other glove types where it produces obvious boundary-driven false positives.
        if glove_type_norm == "latex" and notch:
            best = max(notch, key=lambda d: float(d.score))
            return [
                Defect(
                    label="hole",
                    score=_clamp01(0.52 + 0.36 * float(best.score)),
                    bbox=best.bbox,
                    meta={"source": "tear_notch_fallback"},
                )
            ]
        # Leather tears are often rendered as slit-like dark openings with no missing-mask void.
        # Recover these using a conservative dark-rel + interior-distance detector.
        if glove_type_norm == "leather":
            slit = _detect_leather_slit_tear(bgr, glove_mask_filled)
            if slit:
                return slit
    return out


def _detect_tear_notches(glove_mask_filled: np.ndarray) -> list[Defect]:
    """
    Detect tear-like boundary notches from silhouette geometry.
    This complements `_detect_holes` for edge tears that do not create interior voids.
    """
    mask01 = (glove_mask_filled > 0).astype(np.uint8)
    if mask01.sum() < 450:
        return []
    cnts, _ = cv2.findContours((mask01 * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    contour = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(contour) < 400.0:
        return []

    x, y, w, h = cv2.boundingRect(contour)
    diag = float(np.hypot(w, h))
    if diag < 60.0:
        return []
    top_cut = y + int(round(0.48 * h))
    cuff_cut = y + int(round(0.84 * h))

    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return []
    try:
        defects = cv2.convexityDefects(contour, hull_idx)
    except cv2.error:
        return []
    if defects is None:
        return []

    raw: list[Defect] = []
    for s, e, f, depth in defects.reshape(-1, 4):
        start = contour[int(s)][0].astype(np.float32)
        end = contour[int(e)][0].astype(np.float32)
        far = contour[int(f)][0].astype(np.float32)
        depth_px = float(depth) / 256.0
        span = float(np.linalg.norm(end - start))
        if depth_px < max(10.0, 0.028 * diag):
            continue
        if span < max(6.0, 0.015 * diag) or span > 0.20 * diag:
            continue
        fy = int(far[1])
        if fy <= top_cut:
            continue
        if fy >= cuff_cut:
            continue

        a = np.linalg.norm(end - start) + 1e-6
        b = np.linalg.norm(far - start) + 1e-6
        d = np.linalg.norm(end - far) + 1e-6
        cosang = float((b * b + d * d - a * a) / (2.0 * b * d + 1e-6))
        cosang = max(-1.0, min(1.0, cosang))
        ang = float(np.degrees(np.arccos(cosang)))
        if ang < 14.0 or ang > 95.0:
            continue
        sharpness = float(depth_px / (span + 1e-6))
        if sharpness < 0.34:
            continue

        min_x = int(max(0, min(start[0], end[0], far[0]) - 8))
        min_y = int(max(0, min(start[1], end[1], far[1]) - 8))
        max_x = int(max(start[0], end[0], far[0]) + 8)
        max_y = int(max(start[1], end[1], far[1]) + 8)
        bw = max(8, max_x - min_x)
        bh = max(8, max_y - min_y)
        bbox = BoundingBox(x=min_x, y=min_y, w=bw, h=bh)
        depth_score = min(1.0, depth_px / max(10.0, 0.10 * diag))
        sharp_score = min(1.0, sharpness / 0.70)
        score = 0.45 + 0.30 * depth_score + 0.25 * sharp_score
        raw.append(
            Defect(
                label="tear",
                score=_clamp01(score),
                bbox=bbox,
                meta={"depth_px": float(depth_px), "span": float(span), "sharpness": float(sharpness), "angle": float(ang), "method": "notch"},
            )
        )

    if not raw:
        return []
    raw.sort(key=lambda d: float(d.score), reverse=True)
    kept: list[Defect] = []
    for d in raw:
        if all(_bbox_iou(d.bbox, prev.bbox) < 0.35 for prev in kept):
            kept.append(d)
        if len(kept) >= 2:
            break
    return kept


def _detect_leather_slit_tear(bgr: np.ndarray, glove_mask_filled: np.ndarray) -> list[Defect]:
    """
    Leather-specific slit tear detector for cases where segmentation does not produce an interior void.
    Detects elongated *very dark* openings inside the glove interior by combining:
    - absolute darkness (very low LAB L),
    - local contrast (dark relative to a blurred neighborhood),
    - strong edge evidence around the candidate,
    while enforcing a strict interior-distance constraint to suppress boundary artifacts.
    """
    mf = (glove_mask_filled > 0).astype(np.uint8)
    if int(mf.sum()) < 500:
        return []
    dt = cv2.distanceTransform((mf * 255).astype(np.uint8), cv2.DIST_L2, 5)
    dt_min = 10.0
    interior = (dt >= dt_min).astype(np.uint8)
    if int(interior.sum()) < 300:
        return []

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    vals = L[interior > 0].astype(np.float32).reshape(-1)
    if vals.size < 300:
        return []
    # Use a moderate darkness threshold (low tail of L) and then rely on strong
    # contrast + edge evidence to separate true slit openings from leather wrinkles/stains.
    p2 = float(np.percentile(vals, 2.0))
    p5 = float(np.percentile(vals, 5.0))
    thr_L = float(np.clip(p5 + 4.0, max(16.0, p2 + 2.0), 28.0))

    ys = np.where(mf > 0)[0]
    if ys.size:
        y0, y1 = int(ys.min()), int(ys.max())
        gh = max(1, y1 - y0 + 1)
        cuff_y = y1 - int(round(0.18 * gh))
    else:
        cuff_y = int(mf.shape[0] * 0.85)

    glove_area = float(mf.sum()) + 1e-6
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = cv2.magnitude(gx, gy)

    dark = ((L <= thr_L) & (interior > 0)).astype(np.uint8)
    if int(dark.sum()) < 200:
        return []
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if int(num) <= 1:
        return []

    best: Defect | None = None
    for i in range(1, int(num)):
        x, y, w, h, area = [int(v) for v in stats[i].tolist()]
        if area < 400:
            continue
        cy = y + (h / 2.0)
        if cy >= float(cuff_y):
            continue
        comp = (labels == i).astype(np.uint8)
        area_norm = float(area) / glove_area
        # In this dataset, leather slit tears are visually large interior openings.
        # Enforce a minimum size to avoid confusing small creases / finger-edge shading as tears.
        if area_norm < 0.0080 or area_norm > 0.030:
            continue

        mean_dt = float(dt[comp > 0].mean()) if int(comp.sum()) else 0.0
        if mean_dt < 65.0:
            continue

        cnts, _ = cv2.findContours((comp * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        peri = float(cv2.arcLength(c, True)) + 1e-6
        circ = float((4.0 * np.pi * float(area)) / (peri * peri))
        rect = cv2.minAreaRect(c)
        (rw, rh) = rect[1]
        if rw < 2.0 or rh < 2.0:
            continue
        r_aspect = float(max(rw, rh) / (min(rw, rh) + 1e-6))
        if r_aspect < 4.0:
            continue
        if circ > 0.30:
            continue

        ring = cv2.dilate(comp, np.ones((11, 11), np.uint8), iterations=1) - comp
        ring = (ring > 0).astype(np.uint8)
        ring = (ring & mf).astype(np.uint8)
        if int(ring.sum()) < 100:
            continue
        e_den = float(((ring > 0) & (edges > 0)).sum()) / float(ring.sum() + 1e-6)
        if e_den < 0.15:
            continue
        comp_med = float(np.median(L[comp > 0])) if int(comp.sum()) else 255.0
        ring_med = float(np.median(L[ring > 0])) if int(ring.sum()) else comp_med
        contrast = float(ring_med - comp_med)
        if contrast < 60.0:
            continue
        g_ring = float(np.median(gmag[ring > 0])) if int(ring.sum()) else 0.0
        if g_ring < 70.0:
            continue

        score = _clamp01(
            0.60
            + 0.16 * min(1.0, max(0.0, (r_aspect - 3.5) / 6.0))
            + 0.16 * min(1.0, contrast / 120.0)
            + 0.16 * min(1.0, g_ring / 120.0)
            + 0.12 * min(1.0, mean_dt / 140.0)
            + 0.10 * min(1.0, e_den / 0.35)
        )
        d = Defect(
            label="tear",
            score=score,
            bbox=BoundingBox(x=int(x), y=int(y), w=int(w), h=int(h)),
            meta={
                "source": "leather_slit_abs_dark",
                "L_thr": float(thr_L),
                "L_p2": float(p2),
                "L_p5": float(p5),
                "contrast": float(contrast),
                "g_ring": float(g_ring),
                "r_aspect": float(r_aspect),
                "circularity": float(circ),
                "area_norm": float(area_norm),
                "mean_dt": float(mean_dt),
                "edge_density": float(e_den),
            },
        )
        if best is None or float(d.score) > float(best.score):
            best = d

    return [best] if best is not None else []


def _tear_from_hole_candidates(hole_candidates: list[Defect]) -> list[Defect]:
    """
    Recover tear labels from hole-like blobs when notch extraction misses,
    using shape metadata from `_detect_holes`.
    """
    out: list[Defect] = []
    for d in hole_candidates:
        if str(d.label) != "hole":
            continue
        meta = d.meta or {}
        aspect = float(meta.get("aspect", 1.0))
        circ = float(meta.get("circularity", 1.0))
        area_norm = float(meta.get("area_norm", 0.0))
        is_tear_like = bool(aspect >= 1.45 or circ <= 0.48 or area_norm >= 0.0014)
        if not is_tear_like:
            continue
        tear_score = _clamp01(0.50 + 0.40 * float(d.score) + 0.12 * min(1.0, max(0.0, aspect - 1.2)))
        out.append(
            Defect(
                label="tear",
                score=tear_score,
                bbox=d.bbox,
                meta={"source": "hole_shape_fallback", "aspect": aspect, "circularity": circ, "area_norm": area_norm},
            )
        )
    if not out:
        return []
    out.sort(key=lambda x: float(x.score), reverse=True)
    return out[:2]


def _edge_fold_wrinkle(ctx: DefectDetectionContext) -> list[Defect]:
    """
    Use edge structure to detect fold damage vs wrinkles/dents.
    """
    mask01 = (ctx.glove_mask_filled > 0).astype(np.uint8)
    # Work on interior (avoid boundary edges dominating the score).
    # Using a distance-transform threshold is more stable than dilate/erode rings across segmentation variations.
    dt = cv2.distanceTransform((mask01 * 255).astype(np.uint8), cv2.DIST_L2, 5)
    glove_type = str(ctx.glove_type or "unknown").strip().lower()
    dt_thr = 5.0
    if glove_type == "latex":
        dt_thr = 4.0
    elif glove_type == "leather":
        dt_thr = 5.0
    elif glove_type == "fabric":
        dt_thr = 6.0
    interior = ((mask01 > 0) & (dt >= float(dt_thr))).astype(np.uint8)
    # For fold-line detection, allow a slightly wider band (some folds reach close to the silhouette),
    # but still filter out pure boundary edges later via per-line DT checks.
    dt_thr_hough = float(max(1.5, dt_thr - 1.5))
    if glove_type == "latex":
        dt_thr_hough = 2.5
    elif glove_type == "leather":
        dt_thr_hough = 3.5
    elif glove_type == "fabric":
        dt_thr_hough = 4.5
    interior_hough = ((mask01 > 0) & (dt >= float(dt_thr_hough))).astype(np.uint8)

    edges = (ctx.anomaly.edges * 255).astype(np.uint8)
    edges[interior == 0] = 0

    # Global edge density (in a smoothed sense) indicates wrinkles/texture anomalies.
    density = float(ctx.anomaly.edges[interior > 0].mean()) if interior.sum() else 0.0
    out: list[Defect] = []
    if density < 0.03:
        return out

    fold_density_min = 0.05
    fold_density_max = 0.28
    lsd_aligned_min = 0.48
    lsd_ratio_min = 0.70
    hough_enabled = True
    hough_long_min = 0.34
    hough_dom_min = 0.50
    if glove_type == "leather":
        fold_density_max = 0.24
        lsd_aligned_min = 0.46
        lsd_ratio_min = 0.70
        hough_long_min = 0.38
        hough_dom_min = 0.54
    elif glove_type == "latex":
        fold_density_max = 0.22
        lsd_aligned_min = 0.52
        lsd_ratio_min = 0.74
        hough_enabled = True
        hough_long_min = 0.30
        hough_dom_min = 0.52
    elif glove_type == "fabric":
        fold_density_max = 0.22
        lsd_aligned_min = 0.56
        lsd_ratio_min = 0.76
        hough_enabled = True
        hough_long_min = 0.28
        hough_dom_min = 0.52

    # Detect strong long lines for "damaged by fold"
    gray = cv2.cvtColor(ctx.bgr, cv2.COLOR_BGR2GRAY)
    k = int(max(3, int(ctx.profile.edge_blur_ksize)))
    if (k % 2) == 0:
        k += 1
    gray = cv2.GaussianBlur(gray, (k, k), 0)
    e = cv2.Canny(gray, 60, 160)
    e[interior_hough == 0] = 0
    h, w = mask01.shape[:2]
    max_side = float(max(h, w))
    long_cut = 0.18 * max_side
    if glove_type == "latex":
        long_cut = 0.12 * max_side
    elif glove_type == "fabric":
        long_cut = 0.16 * max_side
    min_len_frac = 0.14
    hough_thresh = 40
    if glove_type == "latex":
        min_len_frac = 0.10
        hough_thresh = 30
    elif glove_type == "fabric":
        min_len_frac = 0.12
        hough_thresh = 35
    min_len = int(max(25, round(min_len_frac * max_side)))
    lines_all = cv2.HoughLinesP(e, 1, np.pi / 180, threshold=int(hough_thresh), minLineLength=int(min_len), maxLineGap=12)
    h_max_wr = 0.0
    h_long_wr = 0.0
    h_total_wr = 0.0
    h_dom_wr = 0.0
    if lines_all is not None and len(lines_all) > 0:
        lengths_wr = []
        angles_wr = []
        for (x1, y1, x2, y2) in lines_all.reshape(-1, 4):
            lengths_wr.append(float(np.hypot(x2 - x1, y2 - y1)))
            ang_wr = float(np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1))))
            ang_wr = (ang_wr + 180.0) % 180.0
            angles_wr.append(ang_wr)
        if lengths_wr:
            arr_wr = np.array(lengths_wr, dtype=np.float32)
            h_max_wr = float(arr_wr.max())
            h_total_wr = float(arr_wr.sum())
            h_long_wr = float(sum(l for l in lengths_wr if l >= float(long_cut)))
            hist_wr, _ = np.histogram(np.array(angles_wr, dtype=np.float32), bins=12, range=(0.0, 180.0), weights=arr_wr)
            h_dom_wr = float(hist_wr.max() / (h_total_wr + 1e-6))

    if density > fold_density_max:
        # For fabric/latex, folds can produce very high edge density; if there is strong
        # long-line evidence, prefer "damaged_by_fold" rather than defaulting to wrinkles.
        if glove_type in {"fabric", "latex"}:
            max_len_req = 0.20 * max_side
            if glove_type == "fabric":
                max_len_req = 0.16 * max_side
            strong_fold_line = bool(
                density >= fold_density_min
                and (
                    (h_dom_wr >= 0.55 and h_max_wr > max_len_req and h_long_wr > 0.22 * max_side)
                    or (h_max_wr > 0.46 * max_side and h_long_wr > 0.55 * max_side)
                    or (glove_type == "fabric" and h_total_wr > 6.0 * max_side and h_max_wr > 0.30 * max_side)
                )
            )
            if strong_fold_line:
                len_score = min(1.0, max(0.0, (h_max_wr / (0.55 * max_side + 1e-6))))
                dom_score = min(1.0, max(0.0, (h_dom_wr - 0.55) / 0.30))
                score = 0.56 + 0.55 * min(1.0, density / 0.26) + 0.10 * len_score + 0.06 * dom_score
                out.append(
                    Defect(
                        label="damaged_by_fold",
                        score=_clamp01(min(0.97, score)),
                        bbox=None,
                        meta={
                            "edge_density": density,
                            "hough_max_len": h_max_wr,
                            "hough_long_sum": h_long_wr,
                            "hough_dom_ratio": h_dom_wr,
                            "method": "hough_density_override",
                        },
                    )
                )
                # Also emit a wrinkle score (folds typically co-occur with strong edge density).
                out.append(
                    Defect(
                        label="wrinkles_dent",
                        score=_clamp01(min(0.88, 0.38 + 1.00 * float(density))),
                        bbox=None,
                        meta={"edge_density": density, "source": "fold_override"},
                    )
                )
                return out
        penalty = 0.0
        if glove_type == "fabric" and h_max_wr > 0.44 * max_side:
            penalty += 0.24 + min(0.24, max(0.0, (h_max_wr / max_side) - 0.44) * 0.65)
        if glove_type == "leather" and h_long_wr > 1.10 * max_side:
            penalty += 0.22 + min(0.24, max(0.0, (h_long_wr / max_side) - 1.10) * 0.11)
        if glove_type == "latex" and h_max_wr > 0.34 * max_side:
            penalty += 0.16 + min(0.16, max(0.0, (h_max_wr / max_side) - 0.34) * 0.45)
        dom_long_min = 1.20 * max_side if glove_type == "fabric" else (1.0 * max_side if glove_type == "leather" else 0.7 * max_side)
        if h_dom_wr >= 0.66 and h_long_wr >= dom_long_min:
            penalty += 0.12
        if glove_type in {"fabric", "leather"} and h_dom_wr >= 0.78 and h_long_wr >= 0.90 * max_side:
            penalty += 0.10 + min(0.10, max(0.0, h_dom_wr - 0.78) * 0.8)
        wr_score = float(min(0.90, 0.40 + 1.05 * float(density)) - penalty)
        if wr_score <= 0.0:
            return out
        out.append(
            Defect(
                label="wrinkles_dent",
                score=_clamp01(wr_score),
                bbox=None,
                meta={
                    "edge_density": density,
                    "reason": "density_too_high_for_fold",
                    "hough_max_len": h_max_wr,
                    "hough_long_sum": h_long_wr,
                    "hough_dom_ratio": h_dom_wr,
                    "penalty": penalty,
                },
            )
        )
        return out

    lsd_lines: list[tuple[float, float]] = []
    try:
        lsd = cv2.createLineSegmentDetector()
        det = lsd.detect(gray)[0]
        if det is not None:
            for seg in det.reshape(-1, 4):
                x1, y1, x2, y2 = [float(v) for v in seg.tolist()]
                mx = int(round((x1 + x2) * 0.5))
                my = int(round((y1 + y2) * 0.5))
                if mx < 0 or mx >= w or my < 0 or my >= h:
                    continue
                if interior[my, mx] == 0:
                    continue
                ln = float(np.hypot(x2 - x1, y2 - y1))
                if ln < 0.16 * max_side:
                    continue
                ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                ang = (ang + 180.0) % 180.0
                lsd_lines.append((ln, ang))
    except Exception:
        lsd_lines = []

    if lsd_lines:
        lengths = np.array([x[0] for x in lsd_lines], dtype=np.float32)
        angles = np.array([x[1] for x in lsd_lines], dtype=np.float32)
        max_len = float(lengths.max())
        total_len = float(lengths.sum())
        hist, bin_edges = np.histogram(angles, bins=12, range=(0.0, 180.0), weights=lengths)
        best_bin = int(np.argmax(hist))
        a0 = float(bin_edges[best_bin])
        a1 = float(bin_edges[best_bin + 1])
        aligned = (angles >= a0) & (angles < a1)
        aligned_len = float(lengths[aligned].sum())
        aligned_ratio = float(aligned_len / (total_len + 1e-6))
        if (
            density >= fold_density_min
            and max_len > 0.24 * max_side
            and aligned_len > lsd_aligned_min * max_side
            and aligned_ratio >= lsd_ratio_min
        ):
            out.append(
                Defect(
                    label="damaged_by_fold",
                    score=_clamp01(min(0.97, 0.52 + 1.00 * float(density))),
                    bbox=None,
                    meta={
                        "edge_density": density,
                        "max_line_len": max_len,
                        "aligned_len": aligned_len,
                        "aligned_ratio": aligned_ratio,
                        "total_len": total_len,
                        "method": "lsd",
                    },
                )
            )
            out.append(
                Defect(
                    label="wrinkles_dent",
                    score=_clamp01(min(0.88, 0.38 + 1.00 * float(density))),
                    bbox=None,
                    meta={"edge_density": density, "source": "fold_lsd"},
                )
            )
            return out

    # Fallback to Hough-based long-line criterion.
    lines = lines_all if hough_enabled else None
    if lines is not None and len(lines) > 0 and density >= fold_density_min:
        lengths = []
        angles = []
        # Reject long boundary-only edges by requiring the line to be interior (median DT along the segment).
        line_dt_min = 4.8 if glove_type == "latex" else 6.0
        for (x1, y1, x2, y2) in lines.reshape(-1, 4):
            xs = np.linspace(float(x1), float(x2), num=9).astype(np.int32)
            ys = np.linspace(float(y1), float(y2), num=9).astype(np.int32)
            xs = np.clip(xs, 0, w - 1)
            ys = np.clip(ys, 0, h - 1)
            med_dt = float(np.median(dt[ys, xs])) if xs.size else 0.0
            if med_dt < float(line_dt_min):
                continue
            lengths.append(float(np.hypot(x2 - x1, y2 - y1)))
            ang = float(np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1))))
            ang = (ang + 180.0) % 180.0
            angles.append(ang)
        if not lengths:
            lines = None
    if lines is not None and len(lines) > 0 and density >= fold_density_min:
        max_len = max(lengths) if lengths else 0.0
        long_sum = float(sum(l for l in lengths if l >= float(long_cut)))
        total_len = float(sum(lengths)) if lengths else 0.0
        if lengths:
            hist, _ = np.histogram(np.array(angles, dtype=np.float32), bins=12, range=(0.0, 180.0), weights=np.array(lengths, dtype=np.float32))
            dom_ratio = float(hist.max() / (total_len + 1e-6))
        else:
            dom_ratio = 0.0
        max_len_min = 0.23 * max_side
        long_sum_min = hough_long_min * max_side
        dom_min = hough_dom_min
        if glove_type == "latex":
            max_len_min = 0.12 * max_side
            long_sum_min = 0.13 * max_side
            dom_min = min(dom_min, 0.50)
        elif glove_type == "fabric":
            max_len_min = 0.20 * max_side
            long_sum_min = min(long_sum_min, 0.22 * max_side)
            dom_min = min(dom_min, 0.52)
        if (max_len > max_len_min and long_sum > long_sum_min and dom_ratio >= dom_min):
            len_score = min(1.0, max(0.0, float(max_len) / float(0.26 * max_side + 1e-6)))
            long_score = min(1.0, max(0.0, float(long_sum) / float(0.40 * max_side + 1e-6)))
            dom_score = min(1.0, max(0.0, float(dom_ratio)))
            fold_score = 0.48 + 1.05 * float(density) + 0.08 * len_score + 0.06 * long_score + 0.04 * dom_score
            out.append(
                Defect(
                    label="damaged_by_fold",
                    score=_clamp01(min(0.95, float(fold_score))),
                    bbox=None,
                    meta={"edge_density": density, "max_line_len": max_len, "long_sum": long_sum, "dom_ratio": dom_ratio, "total_len": total_len, "method": "hough"},
                )
            )
            out.append(
                Defect(
                    label="wrinkles_dent",
                    score=_clamp01(min(0.88, 0.38 + 1.00 * float(density))),
                    bbox=None,
                    meta={"edge_density": density, "source": "fold_hough"},
                )
            )
            return out

    # Otherwise, treat as wrinkles/dent (global texture/edge anomalies).
    out.append(
        Defect(
            label="wrinkles_dent",
            score=_clamp01(min(0.90, 0.40 + 1.05 * float(density))),
            bbox=None,
            meta={"edge_density": density},
        )
    )
    return out


def _spot_stain_discoloration(ctx: DefectDetectionContext) -> list[Defect]:
    mask01 = (ctx.glove_mask > 0).astype(np.uint8)
    glove_type = str(ctx.glove_type or "unknown").strip().lower()
    # Suppress blob defects if segmentation seems to include too much background.
    glove_area = float(mask01.sum())
    h, w = mask01.shape[:2]
    if glove_area < 0.05 * h * w:
        return []
    if int(ctx.quality.get("ok_surface", 0.0)) == 0:
        return []
    x, y, ww, hh = cv2.boundingRect(mask01)
    box_area = float(ww * hh) + 1e-6
    extent = glove_area / box_area
    # If extent is very low, the mask is sparse inside a big box (often background leakage).
    if extent < float(ctx.profile.min_mask_extent_for_surface):
        return []

    cand01, labels, stats = candidate_blobs(
        ctx.anomaly.combined,
        ctx.glove_mask,
        mode=str(ctx.profile.blob_threshold_mode),
        percentile=float(ctx.profile.blob_threshold_percentile),
        offset=float(ctx.profile.blob_threshold_offset),
    )
    if ctx.specular_mask.sum() > 0:
        cand01[ctx.specular_mask > 0] = 0
        num, labels, stats, _ = cv2.connectedComponentsWithStats(cand01, connectivity=8)
        if num <= 1:
            return []
    if stats.shape[0] == 0:
        return []

    lab = cv2.cvtColor(ctx.bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(ctx.bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    vals = lab[mask01 > 0]
    if vals.size == 0:
        return []
    med = np.median(vals, axis=0)
    # Distance-to-boundary to suppress edge-driven artifacts.
    dt = cv2.distanceTransform((mask01 * 255).astype(np.uint8), cv2.DIST_L2, 5)
    ab = lab[:, :, 1:3]
    med_ab = med[1:3].reshape(1, 1, 2)
    de_ab = np.linalg.norm(ab - med_ab, axis=2)
    # Hole/tear voids: pixels inside the filled silhouette that are missing in the raw mask.
    # Spotting is frequently misfired on hole edges; suppress spotting when clear void evidence exists.
    suppress_spotting_for_voids = False
    if glove_type == "leather":
        raw01 = (ctx.glove_mask > 0).astype(np.uint8)
        filled01 = (ctx.glove_mask_filled > 0).astype(np.uint8)
        void01 = ((filled01 > 0) & (raw01 == 0)).astype(np.uint8)
        void_area_frac = float(void01.sum()) / float(filled01.sum() + 1e-6) if int(filled01.sum()) > 0 else 0.0
        if void_area_frac >= 0.0012:
            # Require voids to be interior-ish (avoid small boundary leaks).
            med_dt_void = float(np.median(dt[void01 > 0])) if int(void01.sum()) > 10 else 0.0
            if med_dt_void >= 4.0:
                suppress_spotting_for_voids = True
        # Some segmentations paint over small puncture holes, producing no raw-vs-filled void.
        # Add a content-based puncture check: find background-like pixels inside the filled silhouette.
        if not suppress_spotting_for_voids and int(filled01.sum()) > 800:
            try:
                core = cv2.erode(filled01, np.ones((13, 13), np.uint8), iterations=1)
                if int(core.sum()) < 120:
                    core = filled01
                glove_med = np.median(lab[core > 0], axis=0) if int(core.sum()) > 0 else np.median(lab[filled01 > 0], axis=0)

                ring = cv2.dilate(filled01, np.ones((23, 23), np.uint8), iterations=1) - filled01
                border = np.zeros_like(filled01, dtype=np.uint8)
                b = max(8, int(round(min(filled01.shape[:2]) * 0.06)))
                border[:b, :] = 1
                border[-b:, :] = 1
                border[:, :b] = 1
                border[:, -b:] = 1
                bg_mask = ((ring > 0) | (border > 0)).astype(np.uint8)
                if int(bg_mask.sum()) < 120:
                    bg_mask = (filled01 == 0).astype(np.uint8)
                bg_med = np.median(lab[bg_mask > 0], axis=0) if int(bg_mask.sum()) > 0 else glove_med.copy()

                d_bg = np.linalg.norm(lab - bg_med.reshape(1, 1, 3), axis=2)
                d_gl = np.linalg.norm(lab - glove_med.reshape(1, 1, 3), axis=2)
                bg_like = ((filled01 > 0) & (d_bg <= 12.0) & (d_bg + 3.2 < d_gl)).astype(np.uint8)
                if int(bg_like.sum()) > 0:
                    bg_like = cv2.morphologyEx(bg_like, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
                    bg_like = cv2.morphologyEx(bg_like, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
                    num_v, cc_v, stats_v, _ = cv2.connectedComponentsWithStats(bg_like, connectivity=8)
                    glove_area_local = float(filled01.sum()) + 1e-6
                    for vi in range(1, int(num_v)):
                        x, y, w, h, area = [int(v) for v in stats_v[vi].tolist()]
                        if area < 18:
                            continue
                        area_frac = float(area) / float(glove_area_local)
                        if area_frac < 0.00035:
                            continue
                        blob = (cc_v == vi)
                        med_dt = float(np.median(dt[blob])) if int(area) > 6 else 0.0
                        if med_dt < 6.0:
                            continue
                        # A plausible puncture is compact and not a long seam highlight.
                        aspect = float(max(w, h) / (min(w, h) + 1e-6))
                        if aspect > 3.5 and area > 60:
                            continue
                        suppress_spotting_for_voids = True
                        break
            except Exception:
                pass

    def _spotting_rescue_dark_speckles() -> tuple[int, float, float]:
        """
        Rescue spotting (dark speckles) when anomaly-blob gating misses subtle spots.

        This method:
        - works directly on LAB L-channel local darkness (Gaussian background - L)
        - counts small connected components on an interior-only mask (DT-based)
        - explicitly suppresses segmentation-boundary artifacts by requiring blob DT
        """
        L = (lab[:, :, 0].astype(np.float32) / 255.0).clip(0.0, 1.0)
        glove_area_local = float(mask01.sum()) + 1e-6
        dt_min = 4.6
        k_std = 3.6
        blob_area_max_frac = 0.0016
        max_dim_cap = 30
        aspect_cap = 5.5
        area_cap_px: int | None = None
        if glove_type == "latex":
            dt_min = 5.0
            k_std = 3.8
            blob_area_max_frac = 0.0014
            max_dim_cap = 24
            aspect_cap = 3.6
            area_cap_px = 260
        elif glove_type == "leather":
            dt_min = 4.8
            k_std = 4.2
            blob_area_max_frac = 0.0018
            max_dim_cap = 26
            aspect_cap = 4.2
        elif glove_type == "fabric":
            dt_min = 4.2
            k_std = 4.0
            blob_area_max_frac = 0.0022
            max_dim_cap = 30
            aspect_cap = 4.6

        interior = ((mask01 > 0) & (dt >= float(dt_min))).astype(np.uint8)
        if interior.sum() < 200:
            interior = mask01.copy()

        bg = cv2.GaussianBlur(L, (0, 0), sigmaX=6.0, sigmaY=6.0)
        darkness = (bg - L).clip(0.0, 1.0)
        dvals = darkness[interior > 0]
        if dvals.size < 50:
            return 0, 0.0, 0.0
        base_thr = 0.020
        if glove_type == "leather":
            base_thr = 0.024
        elif glove_type == "fabric":
            base_thr = 0.026
        thr = float(max(base_thr, float(dvals.mean()) + float(k_std) * float(dvals.std())))
        thr = min(0.12, thr)

        cand = ((darkness >= thr) & (interior > 0)).astype(np.uint8)
        if ctx.specular_mask.sum() > 0:
            cand[ctx.specular_mask > 0] = 0
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        num, cc, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)

        count = 0
        area_sum = 0.0
        darkness_sum = 0.0
        blob_area_max = int(round(float(blob_area_max_frac) * glove_area_local))
        blob_area_max = max(18, blob_area_max)
        if area_cap_px is not None:
            blob_area_max = min(int(blob_area_max), int(area_cap_px))
        for i in range(1, int(num)):
            x, y, w, h, area = [int(v) for v in stats[i].tolist()]
            if area < 4:
                continue
            if area > blob_area_max:
                continue
            if max(int(w), int(h)) > int(max_dim_cap):
                continue
            blob = (cc == i).astype(np.uint8)
            if blob.sum() == 0:
                continue
            med_dt = float(np.median(dt[blob > 0]))
            if med_dt < float(dt_min + 0.6):
                continue
            # Avoid counting long edge fragments / wrinkles as "spots".
            aspect = float(max(w, h) / (min(w, h) + 1e-6))
            if aspect > float(aspect_cap) and area > 10:
                continue
            mean_dark = float(darkness[blob > 0].mean())
            if mean_dark < float(thr + 0.004):
                continue
            count += 1
            area_sum += float(area)
            darkness_sum += float(min(0.20, mean_dark))

        area_frac = float(area_sum / glove_area_local)
        avg_dark = float(darkness_sum / max(1, count))
        return int(count), float(area_frac), float(avg_dark)

    def _leather_dense_speckles_evidence() -> tuple[int, float, float, float, float] | None:
        """
        Leather-only dense speckle evidence for `spotting`.

        Uses the same local-darkness speckle extraction as the generic rescue, but also measures:
        - spatial concentration of speckles (top-4 grid cell share)
        - texture anisotropy (structure-tensor coherence median)

        This helps recover the leather spotting FN where anomaly-blob gating collapses due to high edge energy.
        """
        if glove_type != "leather":
            return None
        if suppress_spotting_for_voids:
            return None
        filled01 = (ctx.glove_mask_filled > 0).astype(np.uint8)
        if int(filled01.sum()) < 800:
            return None

        glove_area_local = float(mask01.sum()) + 1e-6
        interior = ((mask01 > 0) & (dt >= 4.8)).astype(np.uint8)
        if int(interior.sum()) < 200:
            interior = mask01.copy()

        L = (lab[:, :, 0].astype(np.float32) / 255.0).clip(0.0, 1.0)
        bg = cv2.GaussianBlur(L, (0, 0), sigmaX=6.0, sigmaY=6.0)
        darkness = (bg - L).clip(0.0, 1.0)
        dvals = darkness[interior > 0]
        if dvals.size < 80:
            return None
        thr = float(max(0.024, float(dvals.mean()) + 4.2 * float(dvals.std())))
        thr = float(min(0.12, thr))

        cand = ((darkness >= thr) & (interior > 0)).astype(np.uint8)
        if ctx.specular_mask.sum() > 0:
            cand[ctx.specular_mask > 0] = 0
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        num, cc, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
        if int(num) <= 1:
            return None

        blob_area_max = int(round(0.0018 * glove_area_local))
        blob_area_max = max(18, blob_area_max)
        max_dim_cap = 26
        aspect_cap = 4.2

        count = 0
        area_sum = 0.0
        darkness_sum = 0.0
        sel_mask = np.zeros_like(mask01, dtype=np.uint8)
        for i in range(1, int(num)):
            x, y, w, h, area = [int(v) for v in stats[i].tolist()]
            if area < 4:
                continue
            if area > blob_area_max:
                continue
            if max(int(w), int(h)) > int(max_dim_cap):
                continue
            aspect = float(max(w, h) / (min(w, h) + 1e-6))
            if aspect > float(aspect_cap) and area > 10:
                continue
            blob = (cc == i).astype(np.uint8)
            if blob.sum() == 0:
                continue
            med_dt = float(np.median(dt[blob > 0]))
            if med_dt < 5.4:
                continue
            mean_dark = float(darkness[blob > 0].mean())
            if mean_dark < float(thr + 0.004):
                continue
            count += 1
            area_sum += float(area)
            darkness_sum += float(min(0.20, mean_dark))
            sel_mask[blob > 0] = 1

        if count <= 0:
            return None
        area_frac = float(area_sum / glove_area_local)
        avg_dark = float(darkness_sum / max(1, count))

        # Speckle concentration: top-4 share over a 6x6 grid in the interior bounding box.
        ys = np.where(interior > 0)[0]
        xs = np.where(interior > 0)[1]
        if ys.size == 0 or xs.size == 0:
            return None
        y0i, y1i = int(ys.min()), int(ys.max())
        x0i, x1i = int(xs.min()), int(xs.max())
        hh = max(1, y1i - y0i + 1)
        ww = max(1, x1i - x0i + 1)
        grid = 6
        spx = float(sel_mask.sum()) + 1e-6
        cells: list[float] = []
        for gy in range(grid):
            for gx in range(grid):
                ya = y0i + int(round(float(gy) * float(hh) / float(grid)))
                yb = y0i + int(round(float(gy + 1) * float(hh) / float(grid)))
                xa = x0i + int(round(float(gx) * float(ww) / float(grid)))
                xb = x0i + int(round(float(gx + 1) * float(ww) / float(grid)))
                cells.append(float(sel_mask[ya:yb, xa:xb].sum()))
        cells_arr = np.array(cells, dtype=np.float32)
        top4 = float(np.sort(cells_arr)[-4:].sum() / spx)

        # Texture anisotropy (structure tensor coherence median) in interior.
        gray = cv2.cvtColor(ctx.bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        Jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigmaX=2.0)
        Jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigmaX=2.0)
        Jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigmaX=2.0)
        tr = Jxx + Jyy
        det = Jxx * Jyy - Jxy * Jxy
        tmp = np.sqrt(np.maximum(0.0, (tr * 0.5) * (tr * 0.5) - det))
        l1 = tr * 0.5 + tmp
        l2 = tr * 0.5 - tmp
        coh = (l1 - l2) / (l1 + l2 + 1e-6)
        mag = np.sqrt(gx * gx + gy * gy)
        interior2 = cv2.erode(filled01, np.ones((17, 17), np.uint8), iterations=1)
        interior2 = (interior2 > 0).astype(np.uint8)
        m = (interior2 > 0) & (mag > 0.03)
        if int(np.count_nonzero(m)) < 500:
            m = (interior2 > 0)
        coh_med = float(np.median(coh[m])) if int(np.count_nonzero(m)) > 20 else 1.0

        return int(count), float(area_frac), float(avg_dark), float(top4), float(coh_med)

    out: list[Defect] = []
    dark_spots = 0
    dark_spot_area = 0.0
    dark_spot_darkness = 0.0
    glove_area = float(mask01.sum()) + 1e-6
    stain_edge_max = 0.24
    stain_local_ab_min = 1.6
    stain_local_ab_max = 7.0
    stain_local_dL_max = -0.045
    discolor_edge_max = 0.30
    discolor_local_ab_min = 1.3
    spot_edge_max = 0.16
    spot_min_darkness = 0.035
    spot_min_count = int(ctx.profile.min_spot_count)
    spot_max_count = 150
    spot_area_min = 0.0008
    spot_area_max = 0.030
    spot_base = 0.58
    blob_min_de = 0.34
    spot_delta_L_thr = -0.06
    spot_local_dL_thr = -0.04
    spot_local_de_thr = 0.12
    if glove_type == "fabric":
        stain_edge_max = 0.30
        stain_local_ab_min = 1.6
        stain_local_ab_max = 5.6
        stain_local_dL_max = -0.055
        discolor_edge_max = 0.26
        discolor_local_ab_min = 1.9
        spot_edge_max = 0.20
        spot_min_darkness = 0.035
        spot_min_count = max(int(ctx.profile.min_spot_count), 40)
        spot_max_count = 180
        spot_area_min = 0.006
        spot_area_max = 0.030
        blob_min_de = 0.30
        spot_delta_L_thr = -0.065
        spot_local_dL_thr = -0.045
        spot_local_de_thr = 0.13
    elif glove_type == "latex":
        stain_edge_max = 0.55
        stain_local_ab_min = 1.8
        stain_local_ab_max = 6.4
        stain_local_dL_max = -0.080
        discolor_edge_max = 0.28
        discolor_local_ab_min = 1.8
        spot_edge_max = 0.22
        spot_min_darkness = 0.035
        spot_min_count = max(int(ctx.profile.min_spot_count), 6)
        spot_max_count = 100
        spot_area_min = 0.0008
        spot_area_max = 0.030
        blob_min_de = 0.30
        spot_delta_L_thr = -0.055
        spot_local_dL_thr = -0.035
        spot_local_de_thr = 0.10
    elif glove_type == "leather":
        stain_edge_max = 0.34
        stain_local_ab_min = 2.2
        stain_local_ab_max = 22.0
        stain_local_dL_max = -0.070
        discolor_edge_max = 0.27
        discolor_local_ab_min = 2.0
        spot_edge_max = 0.22
        spot_min_darkness = 0.035
        spot_min_count = max(int(ctx.profile.min_spot_count), 10)
        spot_max_count = 14
        spot_area_min = 0.0012
        spot_area_max = 0.0026
        spot_base = 0.64

    # Latex-specific plastic contamination (transparent film/glossy patch) can look like specular highlight.
    # Detect via bright+low-saturation blobs with local contrast, away from boundaries.
    if glove_type == "latex":
        s01 = (hsv[:, :, 1] / 255.0).astype(np.float32)
        v01 = (hsv[:, :, 2] / 255.0).astype(np.float32)
        spec_dt_min = 4.2
        spec_v_min = 0.84
        spec_s_max = 0.24
        spec = ((v01 >= spec_v_min) & (s01 <= spec_s_max) & (mask01 > 0) & (dt >= spec_dt_min)).astype(np.uint8)
        if spec.sum() > 0:
            spec = cv2.morphologyEx(spec, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
            spec = cv2.morphologyEx(spec, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
            num_s, labels_s, stats_s, _ = cv2.connectedComponentsWithStats(spec, connectivity=8)
            for si in range(1, int(num_s)):
                xx, yy, ww2, hh2, area2 = [int(v) for v in stats_s[si].tolist()]
                if area2 < 55:
                    continue
                area_norm2 = float(area2) / glove_area
                if area_norm2 > 0.012:
                    continue
                blob2 = (labels_s == si).astype(np.uint8)
                ring2 = cv2.dilate(blob2, np.ones((9, 9), np.uint8), iterations=1) - blob2
                ring2 = (ring2 > 0).astype(np.uint8)
                ring2 = (ring2 & mask01).astype(np.uint8)
                if int(ring2.sum()) < 80:
                    continue
                mean_v = float(v01[blob2 > 0].mean())
                mean_s = float(s01[blob2 > 0].mean())
                ring_v = float(v01[ring2 > 0].mean())
                ring_s = float(s01[ring2 > 0].mean())
                v_boost = float(max(0.0, mean_v - ring_v))
                s_drop = float(max(0.0, ring_s - mean_s))
                edge_strength = float(ctx.anomaly.edges[blob2 > 0].mean()) if blob2.sum() else 0.0
                # Tighten aggressively: in this dataset, true plastic contamination on latex
                # appears as a tiny, low-saturation highlight with strong edge response.
                # This suppresses near-universal seam/wrinkle specular FPs.
                if area_norm2 > 0.0005:
                    continue
                if mean_s > 0.085:
                    continue
                if edge_strength < 0.45:
                    continue
                # Require local contrast so we don't fire on uniform glossy gloves.
                if v_boost < 0.035 and edge_strength < 0.16:
                    continue
                score = _clamp01(
                    min(
                        0.98,
                        0.62
                        + 0.18 * min(1.0, v_boost / 0.12)
                        + 0.10 * min(1.0, s_drop / 0.10)
                        + 0.12 * min(1.0, edge_strength / 0.35)
                        + 0.06 * min(1.0, area_norm2 / 0.004),
                    )
                )
                out.append(
                    Defect(
                        label="plastic_contamination",
                        score=score,
                        bbox=BoundingBox(x=xx, y=yy, w=ww2, h=hh2),
                        meta={
                            "source": "latex_specular_blob",
                            "v_boost": float(v_boost),
                            "s_drop": float(s_drop),
                            "edge_strength": float(edge_strength),
                            "area_norm": float(area_norm2),
                            "mean_s": float(mean_s),
                            "mean_v": float(mean_v),
                        },
                    )
                )

    # Fabric-specific plastic contamination (clear tape/film) often appears as a bright, very-low-saturation
    # patch on top of high-frequency knit texture. The blob-anomaly detector can miss it because the tape
    # is close in chroma to the base fabric; recover via specular-like gating + strict interior constraints.
    if glove_type == "fabric":
        s01 = (hsv[:, :, 1] / 255.0).astype(np.float32)
        v01 = (hsv[:, :, 2] / 255.0).astype(np.float32)
        gray01 = (cv2.cvtColor(ctx.bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0).astype(np.float32)
        hp_abs = np.abs(gray01 - cv2.GaussianBlur(gray01, (9, 9), 0))
        spec = ((v01 >= 0.84) & (s01 <= 0.11) & (mask01 > 0) & (dt >= 4.0)).astype(np.uint8)
        if spec.sum() > 0:
            # Suppress boundary ring and cuff band to avoid edge/cuff highlights.
            ys = np.where(mask01 > 0)[0]
            if ys.size:
                y0, y1 = int(ys.min()), int(ys.max())
                gh = max(1, y1 - y0 + 1)
                cuff_y = y1 - int(round(0.18 * gh))
            else:
                cuff_y = int(mask01.shape[0] * 0.85)
            spec[cuff_y:, :] = 0
            boundary_ring = cv2.dilate(mask01, np.ones((25, 25), np.uint8), iterations=1) - cv2.erode(mask01, np.ones((9, 9), np.uint8), iterations=1)
            spec[boundary_ring > 0] = 0

            spec = cv2.morphologyEx(spec, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
            spec = cv2.morphologyEx(spec, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
            num_s, labels_s, stats_s, _ = cv2.connectedComponentsWithStats(spec, connectivity=8)
            for si in range(1, int(num_s)):
                xx, yy, ww2, hh2, area2 = [int(v) for v in stats_s[si].tolist()]
                # Keep only medium+ blobs; tiny ones are knit highlights.
                if area2 < 520:
                    continue
                area_norm2 = float(area2) / glove_area
                if area_norm2 > 0.030:
                    continue
                blob2 = (labels_s == si).astype(np.uint8)
                ring2 = cv2.dilate(blob2, np.ones((9, 9), np.uint8), iterations=1) - blob2
                ring2 = (ring2 > 0).astype(np.uint8)
                ring2 = (ring2 & mask01).astype(np.uint8)
                if int(ring2.sum()) < 120:
                    continue
                mean_v = float(v01[blob2 > 0].mean())
                mean_s = float(s01[blob2 > 0].mean())
                ring_v = float(v01[ring2 > 0].mean())
                ring_s = float(s01[ring2 > 0].mean())
                v_boost = float(max(0.0, mean_v - ring_v))
                s_drop = float(max(0.0, ring_s - mean_s))
                mean_de = float(ctx.anomaly.color[blob2 > 0].mean()) if blob2.sum() else 0.0
                ring_de = float(ctx.anomaly.color[ring2 > 0].mean()) if ring2.sum() else max(0.0, mean_de - 0.10)
                local_de = float(max(0.0, mean_de - ring_de))
                edge_strength = float(ctx.anomaly.edges[blob2 > 0].mean()) if blob2.sum() else 0.0
                ring_edge = float(ctx.anomaly.edges[ring2 > 0].mean()) if ring2.sum() else edge_strength
                edge_drop = float(max(0.0, ring_edge - edge_strength))
                tex_in = float(np.median(hp_abs[blob2 > 0])) if blob2.sum() else 0.0
                tex_ring = float(np.median(hp_abs[ring2 > 0])) if ring2.sum() else tex_in
                tex_drop = float(max(0.0, tex_ring - tex_in))
                # For tape, edges may be weaker than for latex specular blobs; prefer strong local contrast.
                if local_de < 0.40:
                    continue
                if edge_strength < 0.30:
                    continue
                if edge_drop < 0.10:
                    continue
                # Many fabric FPs are broad lighting highlights: they create a very textured ring and a smooth
                # interior, causing a *huge* edge-drop. True tape tends to have moderate edge-drop and some
                # chroma structure (slight saturation dip) vs its ring.
                if s_drop < 0.018:
                    continue
                if edge_drop > 0.20:
                    continue
                if tex_drop < 0.020:
                    continue
                if v_boost < 0.08:
                    continue
                score = _clamp01(
                    min(
                        0.98,
                        0.60
                        + 0.18 * min(1.0, local_de / 0.55)
                        + 0.14 * min(1.0, v_boost / 0.18)
                        + 0.10 * min(1.0, s_drop / 0.08)
                        + 0.10 * min(1.0, edge_strength / 0.45)
                        + 0.08 * min(1.0, tex_drop / 0.050)
                        + 0.06 * min(1.0, area_norm2 / 0.012),
                    )
                )
                out.append(
                    Defect(
                        label="plastic_contamination",
                        score=score,
                        bbox=BoundingBox(x=xx, y=yy, w=ww2, h=hh2),
                        meta={
                            "source": "fabric_specular_tape",
                            "v_boost": float(v_boost),
                            "s_drop": float(s_drop),
                            "edge_strength": float(edge_strength),
                            "edge_drop": float(edge_drop),
                            "area_norm": float(area_norm2),
                            "mean_s": float(mean_s),
                            "mean_v": float(mean_v),
                            "local_de": float(local_de),
                            "tex_drop": float(tex_drop),
                        },
                    )
                )

    # Additional stain candidates from relative darkness (captures large stains that shift the global median).
    # Uses a bandpass-like "large blur minus small blur" measure on LAB L, with strong boundary/cuff suppression.
    ys = np.where(mask01 > 0)[0]
    if ys.size:
        y0, y1 = int(ys.min()), int(ys.max())
        band_h = max(10, int(round(0.20 * (y1 - y0 + 1))))
        cuff_band = np.zeros_like(mask01, dtype=np.uint8)
        cuff_band[max(y0, y1 - band_h + 1) : y1 + 1, :] = 1
        cuff_band &= mask01
    else:
        cuff_band = np.zeros_like(mask01, dtype=np.uint8)

    erode_sz = max(7, int(round(min(h, w) * 0.018)))
    core = cv2.erode(mask01, np.ones((erode_sz, erode_sz), np.uint8), iterations=1)
    if core.sum() < 120:
        core = mask01.copy()

    L = (lab[:, :, 0] / 255.0).astype(np.float32)
    k_small = max(5, int(round(min(h, w) * 0.010)))
    if (k_small % 2) == 0:
        k_small += 1
    k_large = max(31, int(round(min(h, w) * 0.065)))
    if (k_large % 2) == 0:
        k_large += 1
    k_large = int(min(81, k_large))
    L_small = cv2.GaussianBlur(L, (k_small, k_small), 0)
    L_large = cv2.GaussianBlur(L, (k_large, k_large), 0)
    dark_rel = np.clip(L_large - L_small, 0.0, 1.0)
    gray = (cv2.cvtColor(ctx.bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0).astype(np.float32)
    gray_hp = gray - cv2.GaussianBlur(gray, (9, 9), 0)
    hp = np.abs(gray_hp).astype(np.float32)

    dark_area_min = 0.0035
    dark_dt_min = 3.4
    dark_thr_p = 97.8
    dark_thr_off = 0.010
    dark_thr_min = 0.024
    interior_frac_min = 0.55
    latex_hp_local_min = 0.0
    if glove_type == "latex":
        dark_area_min = 0.0025
        dark_dt_min = 4.2
        dark_thr_p = 98.5
        dark_thr_off = 0.010
        dark_thr_min = 0.028
        interior_frac_min = 0.52
        latex_hp_local_min = 0.008
    elif glove_type == "leather":
        dark_area_min = 0.0020
        dark_dt_min = 3.6
        dark_thr_p = 97.4
        dark_thr_off = 0.010
        dark_thr_min = 0.022
        interior_frac_min = 0.52
    elif glove_type == "fabric":
        dark_area_min = 0.0014
        dark_dt_min = 3.6
        dark_thr_p = 98.2
        dark_thr_off = 0.010
        dark_thr_min = 0.025
        interior_frac_min = 0.55

    dark_vals = dark_rel[core > 0]
    if dark_vals.size:
        dark_thr = float(np.percentile(dark_vals, float(dark_thr_p)) + float(dark_thr_off))
        dark_thr = max(float(dark_thr_min), min(0.45, dark_thr))
        dark_cand = ((dark_rel >= dark_thr) & (mask01 > 0) & (dt >= float(dark_dt_min))).astype(np.uint8)
        if ctx.specular_mask.sum() > 0:
            dark_cand[ctx.specular_mask > 0] = 0
        dark_cand = cv2.morphologyEx(dark_cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        close_k = 9
        close_iters = 2
        if glove_type == "fabric":
            close_k = 13
            close_iters = 2
        dark_cand = cv2.morphologyEx(dark_cand, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8), iterations=int(close_iters))
        num_d, labels_d, stats_d, _ = cv2.connectedComponentsWithStats(dark_cand, connectivity=8)
        for j in range(1, int(num_d)):
            xx, yy, ww2, hh2, area2 = [int(v) for v in stats_d[j].tolist()]
            if area2 < 45:
                continue
            area_norm2 = float(area2) / glove_area
            if area_norm2 < float(dark_area_min) or area_norm2 > 0.35:
                continue
            blob2 = (labels_d == j).astype(np.uint8)
            cuff_frac = float((blob2 & cuff_band).sum()) / float(area2 + 1e-6)
            if cuff_frac > 0.55 and area_norm2 < 0.06:
                continue
            med_dt2 = float(np.median(dt[blob2 > 0])) if blob2.sum() else 0.0
            dt_thr2 = float(dark_dt_min + 1.0)
            interior_frac2 = float(np.mean((dt[blob2 > 0] >= dt_thr2).astype(np.float32))) if blob2.sum() else 0.0
            if area_norm2 < 0.06 and interior_frac2 < float(interior_frac_min):
                continue
            edge_blob2 = float(ctx.anomaly.edges[blob2 > 0].mean()) if blob2.sum() else 0.0
            mean_dark2 = float(dark_rel[blob2 > 0].mean()) if blob2.sum() else 0.0
            mean_de2 = float(ctx.anomaly.color[blob2 > 0].mean()) if blob2.sum() else 0.0
            mean_L2 = float(L[blob2 > 0].mean()) if blob2.sum() else 0.0

            ring2 = cv2.dilate(blob2, np.ones((11, 11), np.uint8), iterations=1) - blob2
            ring2 = (ring2 > 0).astype(np.uint8)
            ring2 = (ring2 & mask01).astype(np.uint8)
            if ctx.specular_mask.sum() > 0:
                ring2 = (ring2 & (1 - (ctx.specular_mask > 0).astype(np.uint8))).astype(np.uint8)
            mean_hp = float(hp[blob2 > 0].mean()) if blob2.sum() else 0.0
            mean_hp_ring = float(hp[ring2 > 0].mean()) if int(ring2.sum()) >= 60 else 0.0
            hp_local = float(max(0.0, mean_hp - mean_hp_ring))
            # Latex gloves often have fold/shadow regions; require local texture increase to call it "dirty".
            if glove_type == "latex" and area_norm2 < 0.030 and hp_local < float(latex_hp_local_min):
                continue
            ring_L2 = float(L[ring2 > 0].mean()) if int(ring2.sum()) >= max(60, int(area2 * 0.40)) else float(np.median(L[core > 0]))
            local_drop2 = float(max(0.0, ring_L2 - mean_L2))

            # Prefer interior, low-edge blobs; still recall-oriented for large stains.
            edge_pen2 = max(0.0, (edge_blob2 - (float(stain_edge_max) + 0.06)) / 0.30)
            score2 = _clamp01(
                min(
                    0.99,
                    0.52
                    + 2.20 * float(mean_dark2)
                    + 0.20 * min(1.0, float(local_drop2) / 0.10)
                    + 0.10 * min(1.0, float(med_dt2) / 10.0)
                    + 0.10 * min(1.0, float(mean_de2) / 0.90)
                    - 0.18 * min(1.0, float(edge_pen2)),
                )
            )
            out.append(
                Defect(
                    label="stain_dirty",
                    score=score2,
                    bbox=BoundingBox(x=xx, y=yy, w=ww2, h=hh2),
                    meta={
                        "source": "dark_rel",
                        "dark_thr": float(dark_thr),
                        "mean_dark": float(mean_dark2),
                        "local_drop": float(local_drop2),
                        "median_dt": float(med_dt2),
                        "interior_frac": float(interior_frac2),
                        "edge_blob": float(edge_blob2),
                        "area_norm": float(area_norm2),
                        "cuff_frac": float(cuff_frac),
                        "hp_local": float(hp_local),
                    },
                )
            )
    for i in range(1, stats.shape[0]):
        x, y, w, h, area = [int(v) for v in stats[i].tolist()]
        if area < 25:
            continue
        area_norm = float(area) / glove_area
        # Skip huge blobs (usually background segmentation imperfections).
        if area > 0.25 * glove_area:
            continue
        blob = (labels == i).astype(np.uint8)
        if int((blob & ctx.specular_mask).sum()) > 0:
            continue
        # Suppress boundary-driven blobs (common on segmentation edges and shadows).
        med_dt_blob = float(np.median(dt[blob > 0])) if blob.sum() else 0.0
        small_blob = bool(area <= 180 and max(w, h) <= 30)
        if small_blob:
            # Spotting/plastic rely on additional cues; allow nearer-to-edge blobs here to avoid recall collapse.
            dt_min_small = 2.4
            if glove_type == "fabric":
                dt_min_small = 2.6
            elif glove_type == "leather":
                dt_min_small = 2.6
            elif glove_type == "latex":
                dt_min_small = 2.4
            if med_dt_blob < float(dt_min_small):
                continue
        else:
            dt_min_large = 3.2
            if glove_type == "latex":
                dt_min_large = 4.2
            elif glove_type == "fabric":
                dt_min_large = 3.6
            elif glove_type == "leather":
                dt_min_large = 3.6
            if med_dt_blob < float(dt_min_large):
                continue
        mean_de = float(ctx.anomaly.color[blob > 0].mean()) if blob.sum() else 0.0
        # Tighten slightly to reduce noisy false positives on textured AI images.
        if mean_de < blob_min_de:
            continue
        bbox = BoundingBox(x=x, y=y, w=w, h=h)

        mean_L = float(lab[:, :, 0][blob > 0].mean()) if blob.sum() else float(med[0])
        delta_L = (mean_L - float(med[0])) / 255.0
        mean_de_ab = float(de_ab[blob > 0].mean()) if blob.sum() else 0.0
        ring = cv2.dilate(blob, np.ones((9, 9), np.uint8), iterations=1) - blob
        ring = (ring > 0).astype(np.uint8)
        ring = (ring & mask01).astype(np.uint8)
        if ctx.specular_mask.sum() > 0:
            ring = (ring & (1 - (ctx.specular_mask > 0).astype(np.uint8))).astype(np.uint8)
        if int(ring.sum()) >= max(35, int(area * 0.5)):
            ring_de = float(ctx.anomaly.color[ring > 0].mean())
            ring_L = float(lab[:, :, 0][ring > 0].mean())
            ring_de_ab = float(de_ab[ring > 0].mean())
        else:
            ring_de = max(0.0, mean_de - 0.10)
            ring_L = float(med[0])
            ring_de_ab = max(0.0, mean_de_ab - 2.0)
        local_de = float(max(0.0, mean_de - ring_de))
        local_de_ab = float(max(0.0, mean_de_ab - ring_de_ab))
        local_dL = float((mean_L - ring_L) / 255.0)
        edge_blob = float(ctx.anomaly.edges[blob > 0].mean()) if blob.sum() else 0.0

        # Blob must differ from its immediate ring, not only from global glove median.
        if area < int(0.030 * glove_area):
            if local_de < 0.10:
                continue
        elif local_de < 0.08:
            continue

        # Heuristic label based on size and brightness change.
        if area <= 180 and max(w, h) <= 30:
            mean_s = float(hsv[:, :, 1][blob > 0].mean() / 255.0) if blob.sum() else 0.0
            mean_v = float(hsv[:, :, 2][blob > 0].mean() / 255.0) if blob.sum() else 0.0
            # Separate dark speckle spotting from bright specular specks.
            if (
                delta_L < float(spot_delta_L_thr)
                and local_dL <= float(spot_local_dL_thr)
                and local_de >= float(spot_local_de_thr)
            ):
                dark_spots += 1
                dark_spot_area += float(area)
                dark_spot_darkness += float(min(0.20, max(0.0, -local_dL)))
            # Plastic contamination tends to be bright/specular with low chroma shift.
            if (
                delta_L > 0.12
                and local_dL >= 0.08
                and local_de >= 0.20
                and mean_v >= 0.70
                and float(ctx.anomaly.edges[blob > 0].mean()) >= float(ctx.profile.plastic_min_edge_strength)
            ):
                if glove_type == "fabric":
                    if not (
                        mean_de_ab >= 2.4
                        and local_de >= 0.22
                        and 0.02 <= mean_s <= 0.10
                        and edge_blob <= 0.67
                        and 1.4 <= (max(w, h) / (min(w, h) + 1e-6)) <= 3.0
                        and area_norm <= 0.00025
                    ):
                        continue
                elif glove_type == "leather":
                    if not (
                        mean_de_ab >= 5.5
                        and local_de >= 0.28
                        and 0.12 <= mean_s <= 0.32
                        and mean_v >= 0.80
                        and edge_blob <= 0.55
                        and (max(w, h) / (min(w, h) + 1e-6)) >= 1.4
                    ):
                        continue
                elif glove_type == "latex":
                    if not (mean_de_ab >= 4.8 and local_de >= 0.25 and mean_s <= 0.22):
                        continue
                else:
                    if not (mean_de_ab >= 2.0 and mean_s <= 0.30):
                        continue
                spec_score = max(0.0, min(1.0, (0.35 - mean_s) / 0.35))
                bright_score = max(0.0, min(1.0, (mean_v - 0.70) / 0.22))
                out.append(
                    Defect(
                        label="plastic_contamination",
                        score=_clamp01(0.62 + 0.18 * min(1.0, local_de / 0.40) + 0.12 * min(1.0, mean_de_ab / 8.0) + 0.08 * bright_score + 0.05 * spec_score),
                        bbox=bbox,
                        meta={
                            "delta_L": float(delta_L),
                            "mean_de_ab": float(mean_de_ab),
                            "local_de": float(local_de),
                            "local_dL": float(local_dL),
                            "mean_s": float(mean_s),
                            "mean_v": float(mean_v),
                            "edge_blob": float(edge_blob),
                            "area_norm": float(area_norm),
                            "bbox_aspect": float(max(w, h) / (min(w, h) + 1e-6)),
                        },
                    )
                )
            continue

        if area >= 0.015 * glove_area:
            # Large area: discoloration vs stain/dirty via brightness direction.
            if delta_L < -0.06:
                # If it's mostly a brightness drop with little chroma shift, it's often just shadow.
                if mean_de_ab < 7.0 and mean_de < 0.65:
                    continue
                if local_dL > stain_local_dL_max:
                    continue
                if local_de_ab < stain_local_ab_min or local_de_ab > stain_local_ab_max or edge_blob > stain_edge_max:
                    continue
                out.append(
                    Defect(
                        label="stain_dirty",
                        score=_clamp01(0.40 + 0.40 * mean_de + 0.16 * min(1.0, local_de / 0.22) + 0.10 * max(0.0, stain_edge_max - edge_blob) / max(1e-6, stain_edge_max)),
                        bbox=bbox,
                        meta={"local_de_ab": float(local_de_ab), "edge_blob": float(edge_blob), "local_dL": float(local_dL)},
                    )
                )
            else:
                # Prefer discoloration when chroma differs (not just bright/dim).
                if local_de_ab < max(1.8, discolor_local_ab_min):
                    continue
                if edge_blob > discolor_edge_max:
                    continue
                chroma_boost = min(0.25, mean_de_ab / 45.0)
                local_boost = min(0.18, local_de_ab / 18.0)
                out.append(
                    Defect(
                        label="discoloration",
                        score=_clamp01(0.48 + 0.30 * mean_de + chroma_boost + local_boost + 0.08 * max(0.0, discolor_edge_max - edge_blob) / max(1e-6, discolor_edge_max)),
                        bbox=bbox,
                        meta={"local_de_ab": float(local_de_ab), "edge_blob": float(edge_blob), "local_dL": float(local_dL)},
                    )
                )
        else:
            # Medium blobs: favor stain when darker; otherwise consider discoloration if chroma differs.
            if delta_L < -0.03:
                if local_dL > stain_local_dL_max:
                    continue
                if local_de_ab < stain_local_ab_min or local_de_ab > stain_local_ab_max or edge_blob > stain_edge_max:
                    continue
                out.append(
                    Defect(
                        label="stain_dirty",
                        score=_clamp01(0.38 + 0.38 * mean_de + 0.18 * min(1.0, local_de / 0.24) + 0.12 * max(0.0, stain_edge_max - edge_blob) / max(1e-6, stain_edge_max)),
                        bbox=bbox,
                        meta={"local_de_ab": float(local_de_ab), "edge_blob": float(edge_blob), "local_dL": float(local_dL)},
                    )
                )
            else:
                if local_de_ab < max(1.3, discolor_local_ab_min):
                    continue
                if edge_blob > discolor_edge_max:
                    continue
                chroma_boost = min(0.22, mean_de_ab / 55.0)
                local_boost = min(0.16, local_de_ab / 16.0)
                out.append(
                    Defect(
                        label="discoloration",
                        score=_clamp01(0.44 + 0.34 * mean_de + chroma_boost + local_boost + 0.08 * max(0.0, discolor_edge_max - edge_blob) / max(1e-6, discolor_edge_max)),
                        bbox=bbox,
                        meta={"local_de_ab": float(local_de_ab), "edge_blob": float(edge_blob), "local_dL": float(local_dL)},
                    )
                )

    # If we saw strong puncture/void evidence, suppress `spotting` entirely for leather.
    # This prevents spot detections from firing on hole/tear edges when segmentation filled the void.
    if glove_type == "leather" and suppress_spotting_for_voids:
        return out

    # Spotting should be multiple dark speckles, not just texture noise.
    def _spot_score(count: int, area_frac: float, avg_dark: float, min_count: int, max_count: int, area_min: float, area_max: float, min_dark: float) -> float | None:
        if not (
            int(count) >= int(min_count)
            and int(count) <= int(max_count)
            and float(area_min) <= float(area_frac) <= float(area_max)
            and float(avg_dark) >= float(min_dark)
        ):
            return None
        count_norm = min(1.0, max(0.0, (int(count) - int(min_count) + 1) / float(max(1, int(min_count)))))
        area_norm = max(0.0, min(1.0, (float(area_frac) - float(area_min)) / max(1e-6, (float(area_max) - float(area_min)))))
        darkness_norm = max(0.0, min(1.0, float(avg_dark) / max(1e-6, float(min_dark) + 0.04)))
        return float(_clamp01(min(0.95, float(spot_base) + 0.22 * float(count_norm) + 0.12 * float(darkness_norm) + 0.08 * float(area_norm))))

    spot_area_frac = float(dark_spot_area / glove_area)
    avg_darkness = float(dark_spot_darkness / max(1, dark_spots))
    best_method = None
    best_score = None
    best_meta: dict[str, float | int | str] = {}

    score_main = _spot_score(dark_spots, spot_area_frac, avg_darkness, int(spot_min_count), int(spot_max_count), float(spot_area_min), float(spot_area_max), float(spot_min_darkness))
    if score_main is not None:
        best_method = "anomaly_blobs"
        best_score = float(score_main)
        best_meta = {"count": int(dark_spots), "area_frac": float(spot_area_frac), "avg_darkness": float(avg_darkness)}

    # Rescue pass: allow a slightly different count/area regime (especially for leather speckle patterns),
    # but enforce interior DT to avoid segmentation-edge artifacts.
    rescue_min_count = int(spot_min_count)
    rescue_max_count = int(spot_max_count)
    rescue_area_min = float(spot_area_min)
    rescue_area_max = float(spot_area_max)
    rescue_min_dark = float(spot_min_darkness)
    allow_rescue = glove_type in {"leather", "fabric"} or (glove_type == "latex" and int(dark_spots) >= 4 and int(dark_spots) < int(spot_min_count))
    if glove_type == "leather":
        rescue_min_count = 8
        rescue_max_count = 90
        rescue_area_min = 0.00025
        rescue_area_max = 0.014
        rescue_min_dark = min(rescue_min_dark, 0.030)
    elif glove_type == "fabric":
        rescue_min_count = max(int(spot_min_count), 35)
        rescue_max_count = max(int(spot_max_count), 220)
        rescue_area_min = min(rescue_area_min, 0.004)
        rescue_area_max = max(rescue_area_max, 0.040)
        rescue_min_dark = min(rescue_min_dark, 0.030)
    elif glove_type == "latex":
        rescue_min_count = 18
        rescue_max_count = 70
        rescue_area_min = 0.0010
        rescue_area_max = 0.007
        rescue_min_dark = min(rescue_min_dark, 0.030)

    if allow_rescue:
        rescue_count, rescue_area_frac, rescue_avg_dark = _spotting_rescue_dark_speckles()
        score_rescue = _spot_score(
            rescue_count,
            rescue_area_frac,
            rescue_avg_dark,
            rescue_min_count,
            rescue_max_count,
            rescue_area_min,
            rescue_area_max,
            rescue_min_dark,
        )
        if score_rescue is not None and glove_type == "latex":
            # Latex: treat rescue as strong, high-confidence evidence, but only when it stays in a
            # speckle-like regime (count/area capped above). Emit a score that clears the UI threshold.
            count_strength = max(0.0, min(1.0, (float(rescue_count) - float(rescue_min_count) + 1.0) / 30.0))
            dark_strength = max(0.0, min(1.0, (float(rescue_avg_dark) - 0.10) / 0.10))
            score_rescue = float(_clamp01(min(0.95, 0.90 + 0.04 * count_strength + 0.03 * dark_strength)))
        if score_rescue is not None and (best_score is None or float(score_rescue) > float(best_score)):
            best_method = "l_darkness_rescue"
            best_score = float(score_rescue)
            best_meta = {"count": int(rescue_count), "area_frac": float(rescue_area_frac), "avg_darkness": float(rescue_avg_dark)}

    # Leather dense-speckle rescue:
    # The leather anomaly-edge map is often high across the glove, collapsing the small-blob spot counter.
    # Recover by using local-darkness speckle density + low anisotropy + low spatial concentration.
    if glove_type == "leather" and not suppress_spotting_for_voids:
        ev = _leather_dense_speckles_evidence()
        if ev is not None:
            c, af, ad, top4, coh_med = ev
            # Add a small lower-bound on coherence to avoid triggering on broad stain textures.
            if (af >= 0.035) and (top4 <= 0.36) and (0.67 <= coh_med <= 0.74) and (c >= 220) and (ad >= 0.12):
                score_dense = float(_clamp01(min(0.95, 0.86 + 0.05 * min(1.0, (af - 0.035) / 0.03) + 0.04 * min(1.0, (0.74 - coh_med) / 0.12))))
                if best_score is None or float(score_dense) > float(best_score):
                    best_method = "leather_dense_speckles"
                    best_score = float(score_dense)
                    best_meta = {"count": int(c), "area_frac": float(af), "avg_darkness": float(ad), "top4": float(top4), "coh_med": float(coh_med)}

    if best_score is not None and best_method is not None:
        meta = dict(best_meta)
        meta["method"] = str(best_method)
        meta["main_count"] = int(dark_spots)
        meta["main_area_frac"] = float(spot_area_frac)
        meta["main_avg_darkness"] = float(avg_darkness)
        out.append(Defect(label="spotting", score=float(best_score), bbox=None, meta=meta))
    return out


def _discoloration_only(ctx: DefectDetectionContext) -> list[Defect]:
    """
    Focused discoloration detector:
    - prioritizes chroma shift (LAB a/b) over brightness-only change
    - suppresses near-boundary blobs and tiny specs
    """
    mask01 = (ctx.glove_mask > 0).astype(np.uint8)
    if mask01.sum() < 250:
        return []
    glove_type = str(ctx.glove_type or "unknown").strip().lower()
    h, w = mask01.shape[:2]
    glove_area = float(mask01.sum()) + 1e-6
    ys = np.where(mask01 > 0)[0]
    if ys.size:
        gy0, gy1 = int(ys.min()), int(ys.max())
        gh = max(1, gy1 - gy0 + 1)
        cuff_cut = int(gy1 - 0.16 * gh)
    else:
        cuff_cut = int(0.84 * h)

    lab = cv2.cvtColor(ctx.bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    erode_sz = max(5, int(round(min(h, w) * 0.012)))
    core = cv2.erode(mask01, np.ones((erode_sz, erode_sz), np.uint8), iterations=1)
    if core.sum() < 80:
        core = mask01.copy()

    core_vals = lab[core > 0]
    if core_vals.size == 0:
        return []
    med = np.median(core_vals, axis=0)

    # Leather gloves often have strong panel-to-panel color variation and highlights.
    # For leather, estimate a "base" chroma (a/b) as the dominant cluster in the core
    # so a single large discoloration patch doesn't drag the median.
    base_ab = med[1:3].copy()
    if glove_type == "leather" and core_vals.shape[0] >= 400:
        try:
            ab_vals = core_vals[:, 1:3].astype(np.float32)
            n = int(ab_vals.shape[0])
            step = max(1, int(round(n / 3000.0)))
            sample = ab_vals[::step].copy()
            if int(sample.shape[0]) >= 200:
                cv2.setRNGSeed(1)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
                _compact, labels_k, centers = cv2.kmeans(
                    sample,
                    2,
                    None,
                    criteria,
                    1,
                    cv2.KMEANS_PP_CENTERS,
                )
                labels_k = labels_k.reshape(-1)
                c0 = int(np.count_nonzero(labels_k == 0))
                c1 = int(np.count_nonzero(labels_k == 1))
                dom = 0 if c0 >= c1 else 1
                dom_vals = sample[labels_k == dom]
                if int(dom_vals.shape[0]) >= 120:
                    base_ab = np.median(dom_vals, axis=0).astype(np.float32)
        except Exception:
            base_ab = med[1:3].copy()

    ab = lab[:, :, 1:3]
    de_ab = np.linalg.norm(ab - base_ab.reshape(1, 1, 2), axis=2)
    # Robust threshold from interior distribution.
    thr_p = 95.0
    if glove_type == "leather":
        # More sensitive on leather (recall prioritized); panel/seam suppression below controls FPs.
        thr_p = 92.0
    thr = float(np.percentile(de_ab[core > 0], thr_p) + 2.0)
    thr = max(3.0, min(20.0, thr))
    cand = ((de_ab >= thr) & (mask01 > 0)).astype(np.uint8)
    if ctx.specular_mask.sum() > 0:
        cand[ctx.specular_mask > 0] = 0
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    dt = cv2.distanceTransform((mask01 * 255).astype(np.uint8), cv2.DIST_L2, 5)
    edge_max = 0.30
    local_ab_min = 1.1
    local_l_abs_max = 0.14
    if glove_type == "fabric":
        edge_max = 0.25
        local_ab_min = 1.4
        local_l_abs_max = 0.12
    elif glove_type == "latex":
        edge_max = 0.27
        local_ab_min = 1.2
        local_l_abs_max = 0.12
    elif glove_type == "leather":
        edge_max = 0.26
        local_ab_min = 1.4
        # Leather discoloration can be significantly brighter/darker than the surrounding area.
        # Keep chroma-shift gating but allow larger local brightness deltas to avoid FNs.
        local_l_abs_max = 0.14

    # Leather: suppress clean panel boundaries (design seams) which frequently produce large ab shifts.
    seam_edge_max = 1.0
    gray_edges = None
    if glove_type == "leather":
        seam_edge_max = 0.11
        gray = cv2.cvtColor(ctx.bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray_edges = (cv2.Canny(gray, 50, 140) > 0).astype(np.uint8)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    out: list[Defect] = []
    for i in range(1, num):
        x, y, ww, hh, area = [int(v) for v in stats[i].tolist()]
        if area < 40:
            continue
        area_frac = float(area) / glove_area
        if area_frac < 0.0025 or area_frac > 0.14:
            continue
        cy = y + (hh / 2.0)
        if cy >= float(cuff_cut) and area_frac < 0.028:
            continue

        blob = (labels == i).astype(np.uint8)
        # Reject boundary-driven color shifts.
        med_dt_blob = float(np.median(dt[blob > 0])) if blob.sum() else 0.0
        if med_dt_blob < 4.0:
            continue
        dt_thr = 4.8
        if glove_type == "latex":
            dt_thr = 5.2
        elif glove_type in {"fabric", "leather"}:
            dt_thr = 4.6
        interior_frac = float(np.mean((dt[blob > 0] >= float(dt_thr)).astype(np.float32))) if blob.sum() else 0.0
        cuff_frac = float(blob[int(cuff_cut) :, :].sum()) / float(area + 1e-6) if int(cuff_cut) < int(h) else 0.0

        mean_de_ab = float(de_ab[blob > 0].mean()) if blob.sum() else 0.0
        # Latex discoloration can be relatively subtle in AI renders; avoid requiring
        # a large margin over the global threshold or we miss true positives.
        de_margin = 0.8
        if glove_type == "latex":
            de_margin = 0.35
        if mean_de_ab < (thr + float(de_margin)):
            continue
        mean_L = float(lab[:, :, 0][blob > 0].mean()) if blob.sum() else float(med[0])
        delta_L_signed = float((mean_L - float(med[0])) / 255.0)
        delta_L = abs(delta_L_signed)
        ring = cv2.dilate(blob, np.ones((9, 9), np.uint8), iterations=1) - blob
        ring = (ring > 0).astype(np.uint8)
        ring = (ring & mask01).astype(np.uint8)
        # For large blobs (common on leather), requiring a ring area >= 0.5*blob area is too strict
        # and causes us to fall back to global medians, which can incorrectly reject true defects.
        ring_min_frac = 0.50
        if glove_type == "leather":
            ring_min_frac = 0.12
        if int(ring.sum()) >= max(40, int(area * ring_min_frac)):
            ring_de_ab = float(de_ab[ring > 0].mean())
            ring_L = float(lab[:, :, 0][ring > 0].mean())
        else:
            ring_de_ab = max(0.0, mean_de_ab - 1.5)
            ring_L = float(med[0])
        local_ab_shift = float(max(0.0, mean_de_ab - ring_de_ab))
        local_dL = float((mean_L - ring_L) / 255.0)
        local_l_abs = float(abs(local_dL))
        edge_blob = float(ctx.anomaly.edges[blob > 0].mean()) if blob.sum() else 0.0
        if glove_type == "leather":
            # Strongly downweight/suppress seam-aligned candidates (panel boundaries / stitching).
            if gray_edges is not None and int(blob.sum()) > 0:
                band = cv2.dilate(blob, np.ones((7, 7), np.uint8), iterations=1) - cv2.erode(blob, np.ones((7, 7), np.uint8), iterations=1)
                band = (band > 0).astype(np.uint8)
                denom = float(band.sum()) + 1e-6
                seam_edge_frac = float((gray_edges & band).sum()) / denom
            else:
                seam_edge_frac = 0.0
            if seam_edge_frac > float(seam_edge_max):
                continue
            # Interior/cuff gating: discoloration on leather is typically mid-surface; avoid cuff-dominant blobs.
            if cuff_frac > 0.55 and area_frac > 0.010:
                continue
            if interior_frac < 0.60:
                continue
        if local_ab_shift < local_ab_min:
            continue
        if local_l_abs > local_l_abs_max:
            continue
        if edge_blob > edge_max:
            continue
        # Strong discoloration should have chroma shift, not just brightness shift.
        score = (
            0.43
            + 0.33 * min(1.0, mean_de_ab / (thr + 8.0))
            + 0.16 * min(1.0, local_ab_shift / 5.0)
            + 0.10 * max(0.0, 0.30 - delta_L - 0.5 * local_l_abs)
            + 0.08 * max(0.0, edge_max - edge_blob) / max(1e-6, edge_max)
        )
        if glove_type == "latex":
            # Latex discoloration positives in this AI dataset can have high absolute chroma distance
            # even when the local-ab margin over the robust threshold is small. Boost strong-ab cases
            # so they clear the tuned per-type threshold without lowering it (avoids new normal FPs).
            if mean_de_ab >= 16.0 and local_ab_shift >= 1.15 and edge_blob <= 0.20:
                boost = min(1.0, max(0.0, (mean_de_ab - 16.0) / 6.0))
                score += 0.05 * float(boost)
        if glove_type == "leather":
            # Leather "light patch" discoloration is common and should not be penalized by the
            # global brightness term (which is referenced to a potentially dark glove median).
            # Boost only when the blob is brighter than the core and still has strong chroma shift.
            if delta_L_signed > 0.22:
                bright = min(1.0, max(0.0, (delta_L_signed - 0.22) / 0.12))
                chroma = min(1.0, local_ab_shift / 4.0)
                score += 0.05 * bright * chroma
        bbox = BoundingBox(x=x, y=y, w=ww, h=hh)
        out.append(
            Defect(
                label="discoloration",
                score=_clamp01(score),
                bbox=bbox,
                meta={
                    "local_ab_shift": float(local_ab_shift),
                    "edge_blob": float(edge_blob),
                    "local_l_abs": float(local_l_abs),
                    "delta_L_signed": float(delta_L_signed),
                    "local_dL": float(local_dL),
                    "median_dt": float(med_dt_blob),
                    "interior_frac": float(interior_frac),
                    "cuff_frac": float(cuff_frac),
                },
            )
        )

    out.sort(key=lambda d: float(d.score), reverse=True)
    return out[:4]


def _stain_from_discoloration(glove_type: str, discoloration_candidates: list[Defect]) -> list[Defect]:
    """
    Recover stain/dirty cases that appear as strong dark discoloration blobs.
    """
    g = str(glove_type or "unknown").strip().lower()
    out: list[Defect] = []

    for d in discoloration_candidates:
        meta = d.meta or {}
        local_ab_shift = float(meta.get("local_ab_shift", 0.0))
        local_l_abs = float(meta.get("local_l_abs", 0.0))
        delta_l_signed = float(meta.get("delta_L_signed", 0.0))
        local_dL = float(meta.get("local_dL", 0.0))
        edge_blob = float(meta.get("edge_blob", 1.0))
        median_dt = float(meta.get("median_dt", 0.0))
        interior_frac = float(meta.get("interior_frac", 0.0))
        cuff_frac = float(meta.get("cuff_frac", 0.0))

        score_min = 0.76
        ab_min = 1.8
        edge_max = 0.20
        dark_global_min = -0.020
        dark_local_min = -0.018
        local_abs_min = 0.065
        dt_min = 4.2
        interior_min = 0.55
        if g == "leather":
            score_min = 0.80
            ab_min = 2.4
            edge_max = 0.16
            dark_global_min = -0.015
            dark_local_min = -0.014
            local_abs_min = 0.060
            dt_min = 4.0
            interior_min = 0.55
        elif g == "latex":
            score_min = 0.74
            ab_min = 1.6
            edge_max = 0.22
            dark_global_min = -0.022
            dark_local_min = -0.020
            local_abs_min = 0.060
            dt_min = 4.8
            interior_min = 0.52
        elif g == "fabric":
            score_min = 0.74
            ab_min = 1.7
            edge_max = 0.22
            dark_global_min = -0.024
            dark_local_min = -0.022
            local_abs_min = 0.060
            dt_min = 4.2
            interior_min = 0.60

        # Strongly suppress segmentation-boundary/cuff artifacts.
        if median_dt < float(dt_min):
            continue
        if interior_frac < float(interior_min):
            continue
        if cuff_frac > 0.55 and float(d.score) < 0.92:
            continue

        dark_evidence = bool(
            (delta_l_signed <= dark_global_min)
            or (local_dL <= dark_local_min)
            or (local_l_abs >= local_abs_min and float(d.score) >= 0.88)
        )
        if not (float(d.score) >= score_min and local_ab_shift >= ab_min and edge_blob <= edge_max and dark_evidence):
            continue
        stain_score = _clamp01(
            min(
                0.98,
                0.70
                + 0.18 * min(1.0, float(d.score))
                + 0.08 * min(1.0, local_ab_shift / 5.0)
                + 0.08 * min(1.0, max(0.0, -local_dL) / 0.08),
            )
        )
        out.append(
            Defect(
                label="stain_dirty",
                score=stain_score,
                bbox=d.bbox,
                meta={
                    "source": "discoloration_fallback",
                    "local_ab_shift": local_ab_shift,
                    "local_l_abs": local_l_abs,
                    "delta_L_signed": delta_l_signed,
                    "local_dL": local_dL,
                    "edge_blob": edge_blob,
                    "median_dt": float(median_dt),
                    "interior_frac": float(interior_frac),
                    "cuff_frac": float(cuff_frac),
                },
            )
        )
    return out


def _cuff_region_mask(glove_mask_filled: np.ndarray) -> np.ndarray:
    """
    Approximate wrist/cuff region as the bottom band of the filled glove mask.

    This is a simple approximation; it becomes much stronger when you keep capture framing consistent.
    """
    mask01f = (glove_mask_filled > 0).astype(np.uint8)
    if mask01f.sum() == 0:
        return mask01f
    ys = np.where(mask01f > 0)[0]
    y0, y1 = int(ys.min()), int(ys.max())
    h = y1 - y0 + 1
    band_h = max(10, int(round(0.18 * h)))
    cuff = np.zeros_like(mask01f, dtype=np.uint8)
    cuff[max(y0, y1 - band_h + 1) : y1 + 1, :] = 1
    cuff &= mask01f
    return cuff


def _roll_and_beading(ctx: DefectDetectionContext) -> list[Defect]:
    """
    Heuristics for cuff/edge abnormalities:
    - improper_roll: high boundary complexity in cuff band
    - incomplete_beading: thin cuff edge (low distance-to-edge thickness) in cuff band
    """
    mask01f = (ctx.glove_mask_filled > 0).astype(np.uint8)
    rot_mask, _rot_deg, m = _rotate_mask_to_upright(mask01f)
    cuff = _cuff_region_mask(rot_mask)
    if cuff.sum() < 120:
        return []

    gray = cv2.cvtColor(ctx.bgr, cv2.COLOR_BGR2GRAY)
    rot_gray = cv2.warpAffine(gray, m, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)
    # Boundary band: dilated minus eroded gives a thickness-normalized boundary indicator.
    dil = cv2.dilate(rot_mask, np.ones((7, 7), np.uint8), iterations=1)
    ero = cv2.erode(rot_mask, np.ones((7, 7), np.uint8), iterations=1)
    boundary = ((dil > 0) & (ero == 0)).astype(np.uint8)
    boundary_cuff = (boundary & cuff).astype(np.uint8)

    # Boundary complexity: edge density near cuff boundary.
    k = int(max(3, int(ctx.profile.edge_blur_ksize)))
    if (k % 2) == 0:
        k += 1
    rot_gray = cv2.GaussianBlur(rot_gray, (k, k), 0)
    edges = cv2.Canny(rot_gray, 60, 160)
    edges = ((edges > 0).astype(np.uint8) & boundary_cuff).astype(np.uint8)
    complexity = float(edges.sum()) / float(boundary_cuff.sum() + 1e-6)

    # Thickness: distance transform inside cuff.
    dt = cv2.distanceTransform((rot_mask * 255).astype(np.uint8), cv2.DIST_L2, 5)
    thickness = float(np.median(dt[cuff > 0])) if cuff.sum() else 0.0
    thickness_all = float(np.median(dt[rot_mask > 0])) if rot_mask.sum() else 1.0
    rel_thickness = thickness / (thickness_all + 1e-6)
    x_counts = cuff.sum(axis=0).astype(np.float32)
    x_non = x_counts[x_counts > 0]
    width_cv = float(np.std(x_non) / (np.mean(x_non) + 1e-6)) if x_non.size > 0 else 0.0

    # Cuff opening jaggedness: variability of the bottom-most cuff edge across x.
    ys: list[int] = []
    for xx in range(int(rot_mask.shape[1])):
        col = np.where((rot_mask[:, xx] > 0) & (cuff[:, xx] > 0))[0]
        if col.size:
            ys.append(int(col.max()))
    if ys:
        ys_arr = np.array(ys, dtype=np.float32)
        cuff_y_std = float(np.std(ys_arr))
    else:
        cuff_y_std = 0.0

    out: list[Defect] = []
    if complexity > 0.12 and width_cv >= 0.22:
        out.append(
            Defect(
                label="improper_roll",
                score=_clamp01(min(0.95, 0.35 + 1.95 * complexity)),
                bbox=None,
                meta={"cuff_complexity": complexity, "rel_thickness": rel_thickness, "width_cv": width_cv, "method": "complexity"},
            )
        )
    # Some improper-roll samples have low edge complexity but a strongly irregular cuff-width profile.
    if (
        complexity <= 0.06
        and (
            (width_cv >= 0.68 and rel_thickness >= 0.24)
            or (width_cv >= 0.58 and rel_thickness >= 0.36)
        )
    ):
        out.append(
            Defect(
                label="improper_roll",
                score=_clamp01(min(0.92, 0.58 + 0.28 * min(1.0, width_cv / 0.80) + 0.14 * min(1.0, rel_thickness / 0.55))),
                bbox=None,
                meta={"cuff_complexity": complexity, "rel_thickness": rel_thickness, "width_cv": width_cv, "method": "profile"},
            )
        )
    if complexity < 0.10 and width_cv >= 0.58 and 0.24 <= rel_thickness <= 0.72:
        out.append(
            Defect(
                label="improper_roll",
                score=_clamp01(min(0.92, 0.56 + 0.30 * min(1.0, width_cv / 0.85) + 0.10 * min(1.0, max(0.0, rel_thickness - 0.24) / 0.40))),
                bbox=None,
                meta={"cuff_complexity": complexity, "rel_thickness": rel_thickness, "width_cv": width_cv, "method": "profile_relaxed"},
            )
        )
    # Incomplete beading tends to be "moderately thin" and relatively smooth/regular in cuff profile.
    # Use a stricter complexity cap to avoid misfiring on unrelated defects (e.g., spotting) that happen
    # to have a thin cuff in the render.
    if complexity < 0.045 and width_cv <= 0.45 and 0.30 <= rel_thickness <= 0.64:
        center = 0.44
        dist = abs(rel_thickness - center)
        reg_score = max(0.0, 1.0 - (width_cv / 0.45))
        thickness_score = max(0.0, 1.0 - (dist / 0.18))
        out.append(
            Defect(
                label="incomplete_beading",
                score=_clamp01(min(0.95, 0.60 + 0.22 * thickness_score + 0.12 * reg_score)),
                bbox=None,
                meta={"rel_thickness": rel_thickness, "cuff_complexity": complexity, "width_cv": width_cv, "cuff_y_std": cuff_y_std, "mode": "regular_thin"},
            )
        )

    # Irregular thin opening: some incomplete-beading renders have a ragged cuff edge with high width variation.
    if (
        (0.026 <= complexity <= 0.080)
        and (0.30 <= rel_thickness <= 0.38)
        and (0.55 <= width_cv <= 0.74)
        and (cuff_y_std >= 24.0)
    ):
        center = 0.36
        dist = abs(rel_thickness - center)
        thickness_score = max(0.0, 1.0 - (dist / 0.12))
        jag_score = min(1.0, max(0.0, (cuff_y_std - 22.0) / 18.0))
        width_score = min(1.0, max(0.0, (width_cv - 0.55) / 0.25))
        score = 0.66 + 0.16 * thickness_score + 0.10 * jag_score + 0.06 * width_score
        out.append(
            Defect(
                label="incomplete_beading",
                score=_clamp01(min(0.95, score)),
                bbox=None,
                meta={"rel_thickness": rel_thickness, "cuff_complexity": complexity, "width_cv": width_cv, "cuff_y_std": cuff_y_std, "mode": "irregular_thin"},
            )
        )
    return out


def _finger_count_anomaly(ctx: DefectDetectionContext) -> list[Defect]:
    """
    Detect missing/extra fingers using silhouette-only structural features.
    We estimate a finger count from the glove mask, and flag anomalies vs 5.
    """
    glove_type = str(ctx.glove_type or "unknown").strip().lower()
    mask = (ctx.glove_mask_filled > 0).astype(np.uint8)
    h0, w0 = mask.shape[:2]
    rot_mask, rot_deg, m = _rotate_mask_to_upright(mask)
    raw_mask = (ctx.glove_mask > 0).astype(np.uint8)
    rot_raw = cv2.warpAffine(raw_mask, m, (w0, h0), flags=cv2.INTER_NEAREST, borderValue=0)
    rot_void = ((rot_mask > 0) & (rot_raw == 0)).astype(np.uint8)
    inv_m = cv2.invertAffineTransform(m)

    count, meta = _estimate_finger_count(ctx.glove_mask_filled)
    if count <= 0:
        return []

    expected = 5
    diff = int(count) - expected
    out: list[Defect] = []
    meta = dict(meta)
    meta.update({"finger_count": int(count), "expected": expected, "rot_deg": float(rot_deg)})

    peaks, prof_s, y0, gh = _profile_peaks(rot_mask)
    if gh <= 0:
        return []
    alt_count = 0
    if glove_type == "fabric":
        alt_peaks, _alt_prof, _alt_y0, _alt_h = _profile_peaks_param(
            rot_mask,
            top_frac=0.62,
            win=5,
            percentile=65.0,
            thresh_floor=0.20,
            min_sep_frac=0.03,
        )
        alt_count = int(len(alt_peaks))
        meta.update(
            {
                "alt_profile_peaks": int(alt_count),
                "alt_profile_top_frac": 0.62,
                "alt_profile_win": 5,
                "alt_profile_percentile": 65.0,
                "alt_profile_min_sep_frac": 0.03,
            }
        )
        polar_peaks, polar_meta = _fingertips_polar_signature(
            rot_mask,
            y0=int(y0),
            gh=int(gh),
            top_frac=0.60,
            bins=360,
            smooth_win=9,
            peak_percentile=78.0,
            peak_floor=0.55,
            min_sep_frac=0.07,
        )
        meta.update({"polar_peaks": int(len(polar_peaks)), "polar_meta": polar_meta})
    top_slice = rot_mask[y0 : y0 + int(round(0.48 * gh)), :]
    top_area_ratio = float(top_slice.sum()) / float(rot_mask.sum() + 1e-6)
    top_prof = _top_profile(rot_mask)
    tip_candidates: list[tuple[int, int]] = []
    for px in peaks:
        x = int(max(0, min(rot_mask.shape[1] - 1, px)))
        ys = np.where(rot_mask[:, x] > 0)[0]
        if ys.size == 0:
            continue
        tip_candidates.append((x, int(ys.min())))

    bbox_hint: BoundingBox | None = None
    short_tip_evidence = False
    gap_evidence = False
    short_tip_drop_px = 0.0
    tip_void_ratio = 0.0
    tip_flat_ratio = 0.0
    tip_plateau_ratio = 0.0
    tip_edge_drop = 0.0
    tip_outlier_z = 0.0
    gap_ratio = 1.0
    tip_spread_ratio = 0.0
    tip_short_ratio = 0.0
    if tip_candidates:
        xs = np.array([p[0] for p in tip_candidates], dtype=np.int32)
        ys_tip = np.array([p[1] for p in tip_candidates], dtype=np.int32)
        ys_tip_f = ys_tip.astype(np.float32)
        tip_spread_ratio = float(np.max(ys_tip_f) - np.min(ys_tip_f)) / float(max(1, gh))
        tip_short_ratio = float(np.max(ys_tip_f) - np.median(ys_tip_f)) / float(max(1, gh))
        step = int(np.median(np.diff(np.sort(xs)))) if xs.size >= 3 else max(20, int(round(rot_mask.shape[1] * 0.08)))

        def _set_short_tip(pick_idx: int, ref_tip_local: float) -> None:
            nonlocal bbox_hint
            nonlocal short_tip_evidence
            nonlocal short_tip_drop_px
            nonlocal tip_void_ratio
            nonlocal tip_flat_ratio
            nonlocal tip_plateau_ratio
            nonlocal tip_edge_drop

            x_c = int(xs[pick_idx])
            y_tip = int(ys_tip[pick_idx])
            short_tip_drop_px = float(max(0, float(y_tip) - float(ref_tip_local)))
            bw = max(18, int(round(0.72 * step)))
            y_top = max(0, int(round(float(ref_tip_local) - 0.04 * gh)))
            y_bot = min(rot_mask.shape[0] - 1, int(round(float(y_tip) + 0.10 * gh)))
            x1 = max(0, x_c - bw // 2)
            y1 = max(0, y_top)
            x2 = min(rot_mask.shape[1], x1 + bw)
            y2 = min(rot_mask.shape[0], max(y1 + 10, y_bot))
            bbox_hint = BoundingBox(x=x1, y=y1, w=max(10, x2 - x1), h=max(10, y2 - y1))
            short_tip_evidence = True
            roi_fg = (rot_mask[y1:y2, x1:x2] > 0)
            roi_void = (rot_void[y1:y2, x1:x2] > 0)
            if roi_fg.any():
                tip_void_ratio = float((roi_void & roi_fg).sum()) / float(roi_fg.sum() + 1e-6)

            win = max(8, int(round(0.34 * step)))
            xa = max(0, x_c - win)
            xb = min(rot_mask.shape[1] - 1, x_c + win)
            prof_win = top_prof[xa : xb + 1]
            valid = prof_win >= 0
            if np.count_nonzero(valid) > 8:
                ys_loc = prof_win[valid].astype(np.float32)
                y_rob = float(np.percentile(ys_loc, 18))
                delta = float(max(2, int(round(0.018 * max(1, gh)))))
                near_tip = ys_loc <= (y_rob + delta)
                tip_flat_ratio = float(np.mean(near_tip))
                tip_plateau_ratio = float(_longest_true_run(near_tip)) / float(max(1, near_tip.size))
                if ys_loc.size >= 4:
                    q = max(1, int(round(0.2 * ys_loc.size)))
                    left_m = float(np.mean(ys_loc[:q]))
                    right_m = float(np.mean(ys_loc[-q:]))
                    tip_edge_drop = float(max(0.0, ((left_m + right_m) * 0.5) - y_rob))

        # Ignore extreme side fingers when searching for anomalous center fingers.
        x_lo = int(np.percentile(xs, 12)) if xs.size >= 3 else int(xs.min())
        x_hi = int(np.percentile(xs, 88)) if xs.size >= 3 else int(xs.max())
        center_idx = [i for i, x in enumerate(xs.tolist()) if x_lo <= int(x) <= x_hi]
        if not center_idx:
            center_idx = list(range(len(xs)))

        ref_tip = int(np.percentile(ys_tip[center_idx], 35))
        # Candidate missing location: a finger tip that is significantly shorter than peers.
        short_idx = [i for i in center_idx if int(ys_tip[i]) > ref_tip + int(0.08 * max(1, gh))]
        y_center = ys_tip[center_idx].astype(np.float32)
        y_med = float(np.median(y_center))
        y_mad = float(np.median(np.abs(y_center - y_med))) + 1.0
        y_z = (ys_tip.astype(np.float32) - y_med) / (y_mad + 1e-6)
        outlier_idx = [
            i
            for i in center_idx
            if float(y_z[i]) >= 2.2 and int(ys_tip[i]) > int(round(y_med + 0.06 * max(1, gh)))
        ]
        if short_idx:
            pick = max(short_idx, key=lambda i: int(ys_tip[i]) - ref_tip)
            tip_outlier_z = float(y_z[pick])
            _set_short_tip(pick, float(ref_tip))
        elif outlier_idx:
            pick = max(outlier_idx, key=lambda i: float(y_z[i]))
            tip_outlier_z = float(y_z[pick])
            _set_short_tip(pick, float(y_med))

    if bbox_hint is None and len(peaks) >= 2:
        peaks_sorted = sorted(int(p) for p in peaks)
        gaps = [(peaks_sorted[i + 1] - peaks_sorted[i], i) for i in range(len(peaks_sorted) - 1)]
        if gaps:
            gap_w, gi = max(gaps, key=lambda t: t[0])
            gap_vals = np.array([g for g, _ in gaps], dtype=np.float32)
            gap_med = float(np.median(gap_vals)) if gap_vals.size else float(gap_w)
            gap_ratio = float(gap_w) / float(gap_med + 1e-6)
            x_c = int(round((peaks_sorted[gi] + peaks_sorted[gi + 1]) / 2.0))
            bw = max(18, int(round(0.65 * gap_w)))
            y_top = max(0, int(y0 + 0.02 * gh))
            y_bot = min(rot_mask.shape[0] - 1, int(y0 + 0.34 * gh))
            bbox_hint = BoundingBox(x=max(0, x_c - bw // 2), y=y_top, w=bw, h=max(10, y_bot - y_top))
            gap_evidence = bool(1.45 <= gap_ratio <= 4.0 and gap_w >= max(16, int(round(rot_mask.shape[1] * 0.06))))

    bbox_orig = _bbox_from_rotated_box(bbox_hint, inv_m, w0, h0) if bbox_hint is not None else None

    p_count = int(meta.get("profile_peaks", 0) or 0)
    h_count = int(meta.get("hull_count", 0) or 0)
    have_both = p_count > 0 and h_count > 0
    strong_missing_count = have_both and p_count <= 4 and h_count <= 4
    single_profile_missing = (p_count == 4 and h_count == 0)
    gh_safe = max(1, gh)
    flat_tip_support = bool(
        short_tip_evidence
        and tip_flat_ratio >= 0.30
        and tip_plateau_ratio >= 0.24
        and tip_edge_drop <= float(0.09 * gh_safe)
    )
    outlier_support = bool(short_tip_evidence and tip_outlier_z >= 2.2 and short_tip_drop_px >= float(0.07 * gh_safe))
    truncated_tip_evidence = bool(flat_tip_support and outlier_support)
    # Geometric fallback for amputated/truncated fingertips where raw-vs-filled "void"
    # evidence is weak (common when the missing area is open to the silhouette boundary).
    short_tip_geom_support = bool(
        short_tip_evidence
        and tip_outlier_z >= 1.8
        and short_tip_drop_px >= float(0.095 * gh_safe)
        and (tip_flat_ratio >= 0.22 or tip_plateau_ratio >= 0.18)
    )
    short_tip_supported = bool(
        short_tip_evidence
        and (
            (short_tip_drop_px >= float(0.08 * gh_safe) and tip_void_ratio >= 0.02)
            or short_tip_geom_support
            or truncated_tip_evidence
        )
    )
    single_profile_support = bool(single_profile_missing and (short_tip_supported or (gap_evidence and tip_spread_ratio >= 0.035)))
    severe_single_profile_missing = bool((not have_both) and h_count == 0 and p_count > 0 and p_count <= 2)
    # Conservative count-only fallback: 4-vs-3 agreement pattern is a frequent
    # signature of one missing/truncated finger on otherwise well-separated gloves.
    count_deficit_support = bool(have_both and p_count == 4 and h_count == 3)
    count_consensus_support = bool(
        have_both
        and p_count == 4
        and h_count == 4
        and (gap_ratio >= 1.20 or short_tip_geom_support)
    )
    profile_hull_conflict_support = bool(
        have_both
        and diff == 0
        and p_count == 3
        and h_count >= 5
        and 0.060 <= tip_spread_ratio <= 0.095
        and 0.035 <= tip_short_ratio <= 0.060
    )
    deep_count_missing = False
    if have_both and min(p_count, h_count) <= 2 and max(p_count, h_count) <= 3 and top_area_ratio >= 0.50:
        if glove_type == "fabric":
            deep_count_missing = bool(tip_spread_ratio >= 0.070)
        elif glove_type == "latex":
            deep_count_missing = False
        elif glove_type == "leather":
            deep_count_missing = bool((p_count <= 2) or (h_count <= 2) or (tip_spread_ratio >= 0.040))
        else:
            deep_count_missing = bool(tip_spread_ratio >= 0.050)
    ambiguous_deep_missing = bool(have_both and p_count == 3 and h_count == 3)
    meta.update(
        {
            "short_tip_drop_px": float(short_tip_drop_px),
            "tip_void_ratio": float(tip_void_ratio),
            "tip_flat_ratio": float(tip_flat_ratio),
            "tip_plateau_ratio": float(tip_plateau_ratio),
            "tip_edge_drop": float(tip_edge_drop),
            "tip_outlier_z": float(tip_outlier_z),
            "flat_tip_support": bool(flat_tip_support),
            "truncated_tip_evidence": bool(truncated_tip_evidence),
            "short_tip_geom_support": bool(short_tip_geom_support),
            "tip_spread_ratio": float(tip_spread_ratio),
            "tip_short_ratio": float(tip_short_ratio),
            "gap_ratio": float(gap_ratio),
            "gap_evidence": bool(gap_evidence),
            "short_tip_supported": bool(short_tip_supported),
            "single_profile_missing": bool(single_profile_missing),
            "single_profile_support": bool(single_profile_support),
            "severe_single_profile_missing": bool(severe_single_profile_missing),
            "count_deficit_support": bool(count_deficit_support),
            "count_consensus_support": bool(count_consensus_support),
            "profile_hull_conflict_support": bool(profile_hull_conflict_support),
            "deep_count_missing": bool(deep_count_missing),
            "ambiguous_deep_missing": bool(ambiguous_deep_missing),
            "top_area_ratio": float(top_area_ratio),
        }
    )

    # Fabric-only extra-fingers rescue:
    # The default robust estimator can undercount finger peaks on knitted gloves.
    # Use two independent "sensitive" cues for fabric:
    # - a looser column-sum peak counter (alt_profile)
    # - a polar contour signature (polar_peaks) restricted to the upper glove
    fabric_alt_profile_support = bool(h_count >= 4 and alt_count >= 6)
    polar_count = int(meta.get("polar_peaks", 0) or 0)
    # Use polar as a rescue only when other counters undercount (helps avoid obvious FPs).
    fabric_polar_support = bool(h_count >= 4 and polar_count >= 6 and alt_count <= 5 and p_count <= 5)
    if glove_type == "fabric" and (fabric_alt_profile_support or fabric_polar_support):
        # Score above the UI threshold (0.78) but keep bounded and deterministic.
        score = 0.82
        if fabric_alt_profile_support:
            score += 0.05 * min(1.0, float(max(0, alt_count - 6)) / 2.0)
        if fabric_polar_support:
            score += 0.05 * min(1.0, float(max(0, polar_count - 6)) / 2.0)
        score += 0.03 * min(1.0, float(max(0, h_count - 4)) / 2.0)
        if fabric_alt_profile_support and fabric_polar_support:
            score += 0.03
        meta2 = dict(meta)
        meta2.update(
            {
                "mode": "fabric_sensitive_extra",
                "fabric_alt_profile_support": bool(fabric_alt_profile_support),
                "fabric_polar_support": bool(fabric_polar_support),
            }
        )
        return [Defect(label="extra_fingers", score=_clamp01(score), bbox=bbox_orig, meta=meta2)]

    if have_both and abs(diff) <= 1:
        # If methods disagree too much, avoid weak +/-1 decisions.
        if abs(p_count - h_count) >= 2 and not profile_hull_conflict_support:
            high_count_side = max(p_count, h_count) >= 6
            if not high_count_side:
                # Latex-only rescue: counters disagree and undercount, but the distance-transform
                # profile over the fingertip region can still show many distinct lobes.
                if glove_type == "latex" and have_both and int(count) == 4 and p_count == 4 and h_count <= 2 and gh > 0:
                    dt = cv2.distanceTransform((rot_mask * 255).astype(np.uint8), cv2.DIST_L2, 5)
                    y_bot = int(y0 + int(round(0.60 * gh)))
                    y_bot = max(int(y0) + 1, min(rot_mask.shape[0], y_bot))
                    prof = dt[int(y0) : int(y_bot), :].max(axis=0)
                    prof_s = cv2.GaussianBlur(prof.reshape(1, -1).astype(np.float32), (9, 1), 0).ravel()
                    mmax = float(np.max(prof_s)) if prof_s.size else 0.0
                    if mmax > 1e-6:
                        thr = 0.45 * mmax
                        min_sep = 22
                        peaks_dt: list[int] = []
                        for i in range(1, int(prof_s.size) - 1):
                            if prof_s[i] >= thr and prof_s[i] > prof_s[i - 1] and prof_s[i] >= prof_s[i + 1]:
                                if peaks_dt and (i - peaks_dt[-1]) < min_sep:
                                    if prof_s[i] > prof_s[peaks_dt[-1]]:
                                        peaks_dt[-1] = i
                                else:
                                    peaks_dt.append(int(i))
                        dt_peaks = int(len(peaks_dt))
                        if dt_peaks >= 6:
                            top = rot_mask[int(y0) : int(y0 + int(round(0.58 * gh))), :]
                            x, y, w, h = (
                                cv2.boundingRect((top > 0).astype(np.uint8))
                                if int(top.sum())
                                else (0, int(y0), rot_mask.shape[1], int(round(0.58 * gh)))
                            )
                            bbox_dt = BoundingBox(x=int(x), y=int(y0), w=int(w), h=int(max(12, h)))
                            bbox_dt_orig = _bbox_from_rotated_box(bbox_dt, inv_m, w0, h0)
                            meta2 = dict(meta)
                            meta2.update({"mode": "latex_dt_peak_rescue", "dt_peaks": int(dt_peaks), "dt_thr_rel": 0.45, "dt_min_sep": int(min_sep)})
                            return [
                                Defect(
                                    label="extra_fingers",
                                    score=_clamp01(0.80 + 0.02 * min(6, max(0, dt_peaks - 6))),
                                    bbox=bbox_dt_orig,
                                    meta=meta2,
                                )
                            ]
                return []

    if diff < 0:
        strong_count_missing = bool(
            have_both and p_count >= 3 and h_count >= 3 and max(p_count, h_count) >= 4 and top_area_ratio >= 0.48
        )
        if abs(diff) >= 2:
            if ambiguous_deep_missing and not (short_tip_supported or gap_evidence):
                return []
            if not (
                short_tip_supported
                or gap_evidence
                or strong_count_missing
                or deep_count_missing
                or severe_single_profile_missing
                or count_deficit_support
            ):
                return []
        # For a single-finger deficit, require stronger structural evidence.
        # This avoids false positives when adjacent fingers are close and peaks merge.
        if abs(diff) == 1:
            if have_both:
                if not (strong_missing_count or truncated_tip_evidence or strong_count_missing):
                    return []
                # Allow small (1-count) estimator disagreement when both methods still
                # indicate a deficit. This is common for cut fingertips where one method
                # undercounts slightly more than the other.
                if abs(p_count - h_count) > 1 and not truncated_tip_evidence:
                    return []
                if not (gap_evidence or short_tip_supported or count_consensus_support or count_deficit_support or strong_count_missing):
                    return []
            else:
                # Allow one-method fallback when structural gap/tip evidence is strong enough.
                if not single_profile_support:
                    return []
        elif abs(diff) >= 2 and not have_both and not severe_single_profile_missing:
            return []

        score = 0.55 + 0.16 * min(3, abs(diff))
        if strong_missing_count:
            score += 0.08
        if single_profile_support:
            score += 0.10
        if severe_single_profile_missing:
            score += 0.02
        if count_deficit_support:
            score += 0.08
        if gap_evidence:
            score += 0.10
        if short_tip_supported:
            score += 0.12
        if strong_count_missing:
            score += 0.10
        out.append(Defect(label="missing_finger", score=_clamp01(score), bbox=bbox_orig, meta=meta))
    elif diff == 0:
        # Detect a clearly truncated center finger even when global count appears unchanged.
        if (short_tip_supported and (single_profile_missing or p_count <= 5)) or profile_hull_conflict_support:
            score = 0.68
            if flat_tip_support:
                score += 0.10
            if truncated_tip_evidence:
                score += 0.08
            if short_tip_drop_px >= float(0.16 * gh_safe):
                score += 0.10
            if profile_hull_conflict_support:
                score += 0.08
            out.append(Defect(label="missing_finger", score=_clamp01(score), bbox=bbox_orig, meta=meta))
    elif diff > 0:
        strong_extra_evidence = bool(p_count >= 6 or h_count >= 6)
        if abs(diff) == 1 and not (gap_evidence or strong_extra_evidence):
            return []
        if have_both and abs(p_count - h_count) >= 3 and not strong_extra_evidence:
            return []
        score = 0.55 + 0.16 * min(3, abs(diff))
        if have_both and p_count >= 6 and h_count >= 6:
            score += 0.08
        if strong_extra_evidence:
            score += 0.10
        out.append(Defect(label="extra_fingers", score=_clamp01(score), bbox=bbox_orig, meta=meta))
    return out


def _inside_out(ctx: DefectDetectionContext) -> list[Defect]:
    """
    Heuristic inside-out detector.

    Inside-out gloves often show more seam/boundary texture on the outside.
    We measure edge density in a narrow band around the glove boundary.
    """
    mask01 = (ctx.glove_mask_filled > 0).astype(np.uint8)
    if mask01.sum() < 300:
        return []

    boundary = cv2.dilate(mask01, np.ones((9, 9), np.uint8), iterations=1) - cv2.erode(mask01, np.ones((9, 9), np.uint8), iterations=1)
    boundary = (boundary > 0).astype(np.uint8)
    interior = cv2.erode(mask01, np.ones((17, 17), np.uint8), iterations=1)
    interior = (interior > 0).astype(np.uint8)
    if interior.sum() < 120:
        return []

    gray = cv2.cvtColor(ctx.bgr, cv2.COLOR_BGR2GRAY)
    k = int(max(3, int(ctx.profile.edge_blur_ksize)))
    if (k % 2) == 0:
        k += 1
    gray = cv2.GaussianBlur(gray, (k, k), 0)
    edge01 = (cv2.Canny(gray, 60, 160) > 0).astype(np.uint8)
    edge_b = float((edge01 & boundary).sum()) / float(boundary.sum() + 1e-6)
    edge_i = float((edge01 & interior).sum()) / float(interior.sum() + 1e-6)
    ratio = float(edge_b / (edge_i + 1e-6))
    glove_type = str(ctx.glove_type or "unknown").strip().lower()
    ratio_ok_balanced = bool(0.95 <= ratio <= 1.11)
    interior_ok_balanced = bool(0.04 <= edge_i <= 0.085)
    boundary_ok_balanced = bool(0.04 <= edge_b <= 0.10)
    ratio_ok_moderate = bool(0.80 <= ratio <= 1.30)
    interior_ok_moderate = bool(0.050 <= edge_i <= 0.120)
    boundary_ok_moderate = bool(0.050 <= edge_b <= 0.120)
    edge_gap_ok_moderate = bool(abs(edge_b - edge_i) <= 0.030)
    ratio_ok_fabric_low = bool(glove_type == "fabric" and 0.22 <= ratio <= 0.48)
    interior_ok_fabric_low = bool(glove_type == "fabric" and 0.15 <= edge_i <= 0.30)
    boundary_ok_fabric_low = bool(glove_type == "fabric" and 0.04 <= edge_b <= 0.10)
    ratio_ok_fabric_hi = bool(glove_type == "fabric" and 1.40 <= ratio <= 2.40)
    interior_ok_fabric_hi = bool(glove_type == "fabric" and 0.020 <= edge_i <= 0.055)
    boundary_ok_fabric_hi = bool(glove_type == "fabric" and 0.045 <= edge_b <= 0.095)
    if (ratio_ok_balanced and interior_ok_balanced and boundary_ok_balanced) or (
        ratio_ok_moderate and interior_ok_moderate and boundary_ok_moderate and edge_gap_ok_moderate
    ) or (
        ratio_ok_fabric_low and interior_ok_fabric_low and boundary_ok_fabric_low
    ) or (
        ratio_ok_fabric_hi and interior_ok_fabric_hi and boundary_ok_fabric_hi
    ):
        mode = "balanced"
        if ratio_ok_fabric_hi and interior_ok_fabric_hi and boundary_ok_fabric_hi:
            ratio_score = min(1.0, max(0.0, (ratio - 1.35) / 0.80))
            interior_score = min(1.0, max(0.0, (0.055 - edge_i) / 0.035))
            boundary_score = min(1.0, max(0.0, (edge_b - 0.045) / 0.05))
            score = 0.58 + 0.17 * ratio_score + 0.12 * interior_score + 0.08 * boundary_score
            mode = "fabric_high_ratio"
        elif ratio_ok_fabric_low and interior_ok_fabric_low and boundary_ok_fabric_low:
            ratio_score = min(1.0, max(0.0, (0.48 - ratio) / 0.26))
            interior_score = min(1.0, max(0.0, (edge_i - 0.15) / 0.15))
            boundary_score = min(1.0, max(0.0, (edge_b - 0.04) / 0.06))
            score = 0.56 + 0.16 * ratio_score + 0.10 * interior_score + 0.08 * boundary_score
            mode = "fabric_low_ratio"
        elif ratio_ok_moderate and interior_ok_moderate and boundary_ok_moderate and edge_gap_ok_moderate:
            ratio_score = max(0.0, 1.0 - min(1.0, abs(ratio - 1.0) / 0.30))
            interior_score = min(1.0, max(0.0, (edge_i - 0.05) / 0.07))
            boundary_score = min(1.0, max(0.0, (edge_b - 0.05) / 0.07))
            score = 0.56 + 0.16 * ratio_score + 0.10 * interior_score + 0.08 * boundary_score
            mode = "moderate_balance"
        else:
            ratio_score = max(0.0, 1.0 - min(1.0, abs(ratio - 1.0) / 0.11))
            interior_score = min(1.0, max(0.0, (edge_i - 0.04) / 0.045))
            boundary_score = min(1.0, max(0.0, (edge_b - 0.04) / 0.06))
            score = 0.60 + 0.18 * ratio_score + 0.12 * interior_score + 0.10 * boundary_score
        return [
            Defect(
                label="inside_out",
                score=_clamp01(score),
                bbox=None,
                meta={
                    "boundary_edge_density": edge_b,
                    "interior_edge_density": edge_i,
                    "edge_ratio": ratio,
                    "mode": mode,
                },
            )
        ]

    # Latex/leather rescue: some inside-out renders show noticeably more boundary texture than interior,
    # but do not meet the "balanced" ratio windows above. Accept a higher boundary-to-interior ratio
    # when both densities remain in a plausible range and the boundary clearly exceeds interior.
    if glove_type == "latex":
        # Tighten the ratio-gap window to avoid misfiring on ordinary speckle/spot patterns.
        high_ratio = bool(1.55 <= ratio <= 2.10)
        dens_ok = bool(0.050 <= edge_b <= 0.105 and 0.024 <= edge_i <= 0.075)
        gap_ok = bool((edge_b - edge_i) >= 0.019)
        if high_ratio and dens_ok and gap_ok:
            ratio_score = min(1.0, max(0.0, (ratio - 1.55) / 0.45))
            gap_score = min(1.0, max(0.0, ((edge_b - edge_i) - 0.019) / 0.030))
            boundary_score = min(1.0, max(0.0, (edge_b - 0.050) / 0.055))
            score = 0.64 + 0.22 * ratio_score + 0.12 * gap_score + 0.06 * boundary_score
            return [
                Defect(
                    label="inside_out",
                    score=_clamp01(score),
                    bbox=None,
                    meta={
                        "boundary_edge_density": edge_b,
                        "interior_edge_density": edge_i,
                        "edge_ratio": ratio,
                        "mode": "high_boundary_ratio_rescue",
                    },
                )
            ]
    if glove_type == "leather":
        # Leather can have strong natural grain; keep a narrow ratio band that matches the missed inside-out sample
        # while rejecting the very high-ratio fold/stain textures.
        high_ratio = bool(1.28 <= ratio <= 1.45)
        dens_ok = bool(0.055 <= edge_b <= 0.090 and 0.035 <= edge_i <= 0.075)
        gap_ok = bool((edge_b - edge_i) >= 0.015)
        if high_ratio and dens_ok and gap_ok:
            ratio_score = min(1.0, max(0.0, (ratio - 1.28) / 0.17))
            gap_score = min(1.0, max(0.0, ((edge_b - edge_i) - 0.015) / 0.025))
            boundary_score = min(1.0, max(0.0, (edge_b - 0.055) / 0.035))
            score = 0.66 + 0.20 * ratio_score + 0.10 * gap_score + 0.06 * boundary_score
            return [
                Defect(
                    label="inside_out",
                    score=_clamp01(score),
                    bbox=None,
                    meta={
                        "boundary_edge_density": edge_b,
                        "interior_edge_density": edge_i,
                        "edge_ratio": ratio,
                        "mode": "high_boundary_ratio_rescue",
                    },
                )
            ]
    # Fabric fallback: with heavy blur (k≈11) the absolute edge densities can be low and noisy.
    # Prefer a weakly balanced boundary/interior pattern with low interior edge density.
    if glove_type == "fabric":
        if edge_i < 0.060 and 0.45 <= ratio <= 1.25 and 0.018 <= edge_b <= 0.070:
            ratio_score = max(0.0, 1.0 - min(1.0, abs(ratio - 0.85) / 0.55))
            interior_score = min(1.0, max(0.0, (0.060 - edge_i) / 0.040))
            boundary_score = min(1.0, max(0.0, (0.070 - edge_b) / 0.055))
            score = 0.52 + 0.18 * ratio_score + 0.12 * interior_score + 0.06 * boundary_score
            return [
                Defect(
                    label="inside_out",
                    score=_clamp01(score),
                    bbox=None,
                    meta={
                        "boundary_edge_density": edge_b,
                        "interior_edge_density": edge_i,
                        "edge_ratio": ratio,
                        "mode": "fabric_low_edge_fallback",
                    },
                )
            ]
    return []


def _nms_by_label(defects: list[Defect], label: str, score_threshold: float = 0.6, nms_threshold: float = 0.35, top_k: int = 15) -> list[Defect]:
    boxed = [d for d in defects if d.label == label and d.bbox is not None and float(d.score) >= float(score_threshold)]
    if not boxed:
        return [d for d in defects if d.label != label]

    boxes = [[int(d.bbox.x), int(d.bbox.y), int(d.bbox.w), int(d.bbox.h)] for d in boxed]
    scores = [float(d.score) for d in boxed]
    idxs = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=float(score_threshold), nms_threshold=float(nms_threshold))
    keep = set()
    if len(idxs) > 0:
        flat = [int(i) for i in np.array(idxs).reshape(-1).tolist()]
        # sort by score desc
        flat.sort(key=lambda i: scores[i], reverse=True)
        for i in flat[: int(top_k)]:
            keep.add(i)

    kept = [boxed[i] for i in range(len(boxed)) if i in keep]
    others = [d for d in defects if d.label != label]
    return others + kept


def detect_defects(
    bgr: np.ndarray,
    glove_mask: np.ndarray,
    glove_mask_filled: np.ndarray,
    glove_type: str = "unknown",
    focus_only: bool = False,
    allowed_labels: set[str] | None = None,
) -> tuple[list[Defect], AnomalyMaps]:
    glove_type_norm = str(glove_type or "unknown").strip().lower()
    profile = get_defect_profile(glove_type_norm)
    req: set[str] | None = None
    if allowed_labels is not None:
        req = set(str(x) for x in allowed_labels)
    if focus_only:
        req = set(FOCUS_LABELS) if req is None else (req & set(FOCUS_LABELS))
    # Cuff defects are only meaningful for latex gloves.
    if req is not None and glove_type_norm != "latex":
        req = req - LABEL_GROUP_CUFF

    # Decide whether we need anomaly maps (expensive) based on requested labels.
    need_anom = True if req is None else bool(req & LABELS_NEED_ANOMALY)
    specular_mask = _specular_mask01(bgr, glove_mask) if bool(profile.enable_specular_suppression) else np.zeros(glove_mask.shape[:2], dtype=np.uint8)
    anomaly = (
        build_anomaly_maps(
            bgr,
            glove_mask,
            color_weight=float(profile.anomaly_color_weight),
            texture_weight=float(profile.anomaly_texture_weight),
            edge_weight=float(profile.anomaly_edge_weight),
            edge_blur_ksize=int(profile.edge_blur_ksize),
            suppress_mask01=specular_mask if bool(profile.enable_specular_suppression) else None,
        )
        if need_anom
        else _zero_anomaly_like(glove_mask)
    )
    quality = _mask_quality(glove_mask, bgr)
    ctx = DefectDetectionContext(
        bgr=bgr,
        glove_mask=glove_mask,
        glove_mask_filled=glove_mask_filled,
        anomaly=anomaly,
        glove_type=glove_type_norm,
        profile=profile,
        specular_mask=specular_mask,
        quality=quality,
    )

    defects: list[Defect] = []
    allow_surface = bool(int(ctx.quality.get("ok_surface", 0.0)) == 1)
    allow_fingers = bool(int(ctx.quality.get("ok_finger", 0.0)) == 1)
    # Compute only the detector groups needed for the requested label set.
    if req is None:
        # "All labels" mode.
        hole_like = _detect_holes(bgr, glove_mask, glove_mask_filled, glove_type=glove_type_norm)
        defects.extend(hole_like)
        if not any(d.label == "tear" for d in hole_like):
            tear_like = _tear_from_hole_candidates(hole_like)
            defects.extend(tear_like)
        if allow_surface:
            surf = _spot_stain_discoloration(ctx)
            surf = [d for d in surf if d.label != "discoloration"]
            defects.extend(surf)
            discol = _discoloration_only(ctx)
            defects.extend(_stain_from_discoloration(glove_type_norm, discol))
            defects.extend(discol)
        defects.extend(_edge_fold_wrinkle(ctx))
        if glove_type_norm == "latex":
            defects.extend(_roll_and_beading(ctx))
        if allow_fingers:
            defects.extend(_finger_count_anomaly(ctx))
        defects.extend(_inside_out(ctx))
    else:
        if req & LABEL_GROUP_HOLES:
            holes = _detect_holes(bgr, glove_mask, glove_mask_filled, glove_type=glove_type_norm)
            if "tear" in req and not any(d.label == "tear" for d in holes):
                tear_like = _tear_from_hole_candidates(holes)
                holes.extend(tear_like)
            # Keep only the requested subset (hole/tear) for this detector.
            holes = [d for d in holes if d.label in req]
            defects.extend(holes)

        discol_for_req: list[Defect] = []
        if allow_surface and (("discoloration" in req) or ("stain_dirty" in req)):
            # Discoloration has a dedicated detector that doesn't require anomaly maps.
            discol_for_req = _discoloration_only(ctx)
            if "discoloration" in req:
                defects.extend(discol_for_req)
            if "stain_dirty" in req:
                defects.extend(_stain_from_discoloration(glove_type_norm, discol_for_req))

        if (req & LABEL_GROUP_SURFACE) and allow_surface:
            surf = _spot_stain_discoloration(ctx)
            surf = [d for d in surf if d.label in req]
            defects.extend(surf)

        if req & LABEL_GROUP_WRINKLE:
            wr = _edge_fold_wrinkle(ctx)
            wr = [d for d in wr if d.label in req]
            defects.extend(wr)

        if (req & LABEL_GROUP_CUFF) and glove_type_norm == "latex":
            cuff = _roll_and_beading(ctx)
            cuff = [d for d in cuff if d.label in req]
            defects.extend(cuff)

        if (req & LABEL_GROUP_FINGERS) and allow_fingers:
            fc = _finger_count_anomaly(ctx)
            fc = [d for d in fc if d.label in req]
            defects.extend(fc)

        if "inside_out" in req:
            defects.extend(_inside_out(ctx))

    # De-duplicate by label (keep max score) but keep bboxes for localized defects.
    best: dict[str, Defect] = {}
    localized = {"tear", "hole", "discoloration", "stain_dirty", "plastic_contamination"}
    for d in defects:
        if d.label in localized:
            # Keep multiple instances for localized labels.
            key = f"{d.label}:{d.bbox.x if d.bbox else -1}:{d.bbox.y if d.bbox else -1}"
            best[key] = d
        else:
            prev = best.get(d.label)
            if prev is None or d.score > prev.score:
                best[d.label] = d

    # Return in a stable order: higher scores first.
    out = list(best.values())
    out.sort(key=lambda x: float(x.score), reverse=True)

    # Reduce clutter by applying NMS on common blob labels.
    for lab in ("stain_dirty", "discoloration", "plastic_contamination"):
        out = _nms_by_label(out, lab, score_threshold=0.65, nms_threshold=0.35, top_k=12)
        out.sort(key=lambda x: float(x.score), reverse=True)

    # Suppress boundary-adjacent "plastic contamination" boxes (common seam/edge artifacts).
    # Keep recall-first by using a low interior threshold; true plastic/tape blobs in this dataset
    # are deep in the glove interior.
    if any(d.label == "plastic_contamination" and d.bbox is not None for d in out):
        mf = (glove_mask_filled > 0).astype(np.uint8)
        if int(mf.sum()) > 250:
            dt = cv2.distanceTransform((mf * 255).astype(np.uint8), cv2.DIST_L2, 5)
            kept: list[Defect] = []
            for d in out:
                if d.label != "plastic_contamination" or d.bbox is None:
                    kept.append(d)
                    continue
                x1, y1, x2, y2 = d.bbox.as_xyxy()
                h, w = mf.shape[:2]
                x1 = max(0, min(w - 1, int(x1)))
                y1 = max(0, min(h - 1, int(y1)))
                x2 = max(0, min(w, int(x2)))
                y2 = max(0, min(h, int(y2)))
                if x2 <= x1 or y2 <= y1:
                    continue
                sub_m = mf[y1:y2, x1:x2]
                if int(sub_m.sum()) < 10:
                    continue
                vals = dt[y1:y2, x1:x2][sub_m > 0]
                if vals.size <= 0:
                    continue
                med_dt = float(np.median(vals))
                interior_frac = float(np.mean((vals >= 4.0).astype(np.float32)))
                if med_dt < 6.0 or interior_frac < 0.55:
                    continue
                kept.append(d)
            out = kept

    if allowed_labels is not None:
        allowed = set(str(x) for x in allowed_labels)
        out = [d for d in out if d.label in allowed]
        # For "one defect at a time" UX, keep only the best hole candidate when the user
        # explicitly requests just `hole` (avoids clutter from segmentation artifacts).
        if allowed == {"hole"}:
            out.sort(key=lambda x: float(x.score), reverse=True)
            out = out[:1]
    return out, anomaly
