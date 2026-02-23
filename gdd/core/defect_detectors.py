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
        if notch:
            best = max(notch, key=lambda d: float(d.score))
            return [
                Defect(
                    label="hole",
                    score=_clamp01(0.52 + 0.36 * float(best.score)),
                    bbox=best.bbox,
                    meta={"source": "tear_notch_fallback"},
                )
            ]
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
    boundary = cv2.dilate(mask01, np.ones((9, 9), np.uint8), iterations=1) - cv2.erode(mask01, np.ones((9, 9), np.uint8), iterations=1)
    boundary = (boundary > 0).astype(np.uint8)
    interior = (mask01 & (1 - cv2.dilate(boundary, np.ones((13, 13), np.uint8), iterations=1))).astype(np.uint8)

    edges = (ctx.anomaly.edges * 255).astype(np.uint8)
    edges[interior == 0] = 0

    # Global edge density (in a smoothed sense) indicates wrinkles/texture anomalies.
    density = float(ctx.anomaly.edges[interior > 0].mean()) if interior.sum() else 0.0
    out: list[Defect] = []
    if density < 0.03:
        return out

    glove_type = str(ctx.glove_type or "unknown").strip().lower()
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
        hough_enabled = False
    elif glove_type == "fabric":
        fold_density_max = 0.22
        lsd_aligned_min = 0.56
        lsd_ratio_min = 0.76
        hough_enabled = False

    # Detect strong long lines for "damaged by fold"
    gray = cv2.cvtColor(ctx.bgr, cv2.COLOR_BGR2GRAY)
    k = int(max(3, int(ctx.profile.edge_blur_ksize)))
    if (k % 2) == 0:
        k += 1
    gray = cv2.GaussianBlur(gray, (k, k), 0)
    e = cv2.Canny(gray, 60, 160)
    e[interior == 0] = 0
    h, w = mask01.shape[:2]
    max_side = float(max(h, w))
    min_len = int(max(35, round(0.14 * max_side)))
    lines_all = cv2.HoughLinesP(e, 1, np.pi / 180, threshold=40, minLineLength=min_len, maxLineGap=12)
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
            h_long_wr = float(sum(l for l in lengths_wr if l >= 0.18 * max_side))
            hist_wr, _ = np.histogram(np.array(angles_wr, dtype=np.float32), bins=12, range=(0.0, 180.0), weights=arr_wr)
            h_dom_wr = float(hist_wr.max() / (h_total_wr + 1e-6))

    if density > fold_density_max:
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
            return out

    # Fallback to Hough-based long-line criterion.
    lines = lines_all if hough_enabled else None
    if lines is not None and len(lines) > 0 and density >= fold_density_min:
        lengths = []
        angles = []
        for (x1, y1, x2, y2) in lines.reshape(-1, 4):
            lengths.append(float(np.hypot(x2 - x1, y2 - y1)))
            ang = float(np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1))))
            ang = (ang + 180.0) % 180.0
            angles.append(ang)
        max_len = max(lengths) if lengths else 0.0
        long_sum = float(sum(l for l in lengths if l >= 0.18 * max_side))
        total_len = float(sum(lengths)) if lengths else 0.0
        if lengths:
            hist, _ = np.histogram(np.array(angles, dtype=np.float32), bins=12, range=(0.0, 180.0), weights=np.array(lengths, dtype=np.float32))
            dom_ratio = float(hist.max() / (total_len + 1e-6))
        else:
            dom_ratio = 0.0
        if (
            max_len > 0.23 * max_side
            and long_sum > hough_long_min * max_side
            and dom_ratio >= hough_dom_min
        ):
            out.append(
                Defect(
                    label="damaged_by_fold",
                    score=_clamp01(min(0.95, 0.48 + 1.05 * float(density))),
                    bbox=None,
                    meta={"edge_density": density, "max_line_len": max_len, "long_sum": long_sum, "dom_ratio": dom_ratio, "total_len": total_len, "method": "hough"},
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
    if glove_type == "fabric":
        stain_edge_max = 0.18
        stain_local_ab_min = 2.0
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
    elif glove_type == "latex":
        stain_edge_max = 0.10
        stain_local_ab_min = 2.6
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
    elif glove_type == "leather":
        stain_edge_max = 0.20
        stain_local_ab_min = 3.0
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
        if blob.sum():
            if float(np.median(dt[blob > 0])) < 3.2:
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
                delta_L < -0.06
                and local_dL <= -0.04
                and local_de >= 0.12
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

    # Spotting should be multiple dark speckles, not just texture noise.
    spot_area_frac = float(dark_spot_area / glove_area)
    avg_darkness = float(dark_spot_darkness / max(1, dark_spots))
    if (
        dark_spots >= int(spot_min_count)
        and dark_spots <= int(spot_max_count)
        and spot_area_min <= spot_area_frac <= spot_area_max
        and avg_darkness >= spot_min_darkness
    ):
        count_norm = min(1.0, max(0.0, (dark_spots - int(spot_min_count) + 1) / float(max(1, int(spot_min_count)))))
        area_norm = max(0.0, min(1.0, (spot_area_frac - spot_area_min) / max(1e-6, (spot_area_max - spot_area_min))))
        darkness_norm = max(0.0, min(1.0, avg_darkness / max(1e-6, spot_min_darkness + 0.04)))
        out.append(
            Defect(
                label="spotting",
                score=_clamp01(min(0.95, spot_base + 0.22 * count_norm + 0.12 * darkness_norm + 0.08 * area_norm)),
                bbox=None,
                meta={"count": dark_spots, "area_frac": float(spot_area_frac), "avg_darkness": float(avg_darkness)},
            )
        )
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

    ab = lab[:, :, 1:3]
    de_ab = np.linalg.norm(ab - med[1:3].reshape(1, 1, 2), axis=2)
    # Robust threshold from interior distribution.
    thr = float(np.percentile(de_ab[core > 0], 95) + 2.0)
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
        local_l_abs_max = 0.14
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
        if float(np.median(dt[blob > 0])) < 4.0:
            continue

        mean_de_ab = float(de_ab[blob > 0].mean()) if blob.sum() else 0.0
        if mean_de_ab < (thr + 0.8):
            continue
        mean_L = float(lab[:, :, 0][blob > 0].mean()) if blob.sum() else float(med[0])
        delta_L_signed = float((mean_L - float(med[0])) / 255.0)
        delta_L = abs(delta_L_signed)
        ring = cv2.dilate(blob, np.ones((9, 9), np.uint8), iterations=1) - blob
        ring = (ring > 0).astype(np.uint8)
        ring = (ring & mask01).astype(np.uint8)
        if int(ring.sum()) >= max(40, int(area * 0.5)):
            ring_de_ab = float(de_ab[ring > 0].mean())
            ring_L = float(lab[:, :, 0][ring > 0].mean())
        else:
            ring_de_ab = max(0.0, mean_de_ab - 1.5)
            ring_L = float(med[0])
        local_ab_shift = float(max(0.0, mean_de_ab - ring_de_ab))
        local_dL = float((mean_L - ring_L) / 255.0)
        local_l_abs = float(abs(local_dL))
        edge_blob = float(ctx.anomaly.edges[blob > 0].mean()) if blob.sum() else 0.0
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

        score_min = 0.76
        ab_min = 1.8
        edge_max = 0.20
        dark_global_min = -0.020
        dark_local_min = -0.018
        local_abs_min = 0.065
        if g == "leather":
            score_min = 0.80
            ab_min = 2.4
            edge_max = 0.16
            dark_global_min = -0.015
            dark_local_min = -0.014
            local_abs_min = 0.060
        elif g == "latex":
            score_min = 0.74
            ab_min = 1.6
            edge_max = 0.22
            dark_global_min = -0.022
            dark_local_min = -0.020
            local_abs_min = 0.060
        elif g == "fabric":
            score_min = 0.74
            ab_min = 1.7
            edge_max = 0.22
            dark_global_min = -0.024
            dark_local_min = -0.022
            local_abs_min = 0.060

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
    if complexity < 0.11 and width_cv <= 0.45 and 0.30 <= rel_thickness <= 0.64:
        center = 0.44
        dist = abs(rel_thickness - center)
        reg_score = max(0.0, 1.0 - (width_cv / 0.45))
        thickness_score = max(0.0, 1.0 - (dist / 0.18))
        out.append(
            Defect(
                label="incomplete_beading",
                score=_clamp01(min(0.95, 0.60 + 0.22 * thickness_score + 0.12 * reg_score)),
                bbox=None,
                meta={"rel_thickness": rel_thickness, "cuff_complexity": complexity, "width_cv": width_cv},
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

    if have_both and abs(diff) <= 1:
        # If methods disagree too much, avoid weak +/-1 decisions.
        if abs(p_count - h_count) >= 2 and not profile_hull_conflict_support:
            high_count_side = max(p_count, h_count) >= 6
            if not high_count_side:
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
            tear_like = _detect_tear_notches(glove_mask_filled)
            if not tear_like:
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
                tear_like = _detect_tear_notches(glove_mask_filled)
                if not tear_like:
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

    if allowed_labels is not None:
        allowed = set(str(x) for x in allowed_labels)
        out = [d for d in out if d.label in allowed]
        # For "one defect at a time" UX, keep only the best hole candidate when the user
        # explicitly requests just `hole` (avoids clutter from segmentation artifacts).
        if allowed == {"hole"}:
            out.sort(key=lambda x: float(x.score), reverse=True)
            out = out[:1]
    return out, anomaly
