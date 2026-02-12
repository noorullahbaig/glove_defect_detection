from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .anomaly import AnomalyMaps, build_anomaly_maps, candidate_blobs
from .types import BoundingBox, Defect


@dataclass(frozen=True)
class DefectDetectionContext:
    bgr: np.ndarray
    glove_mask: np.ndarray
    glove_mask_filled: np.ndarray
    anomaly: AnomalyMaps


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

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
    defects = cv2.convexityDefects(c, hull)
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


def _detect_holes(bgr: np.ndarray, glove_mask: np.ndarray, glove_mask_filled: np.ndarray) -> list[Defect]:
    """
    Detect holes/tears by finding regions that are inside the filled silhouette but missing in the raw mask.
    """
    m = (glove_mask > 0).astype(np.uint8)
    mf = (glove_mask_filled > 0).astype(np.uint8)
    glove_area = float(mf.sum()) + 1e-6
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

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
    if holes.sum() == 0:
        return []
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    out: list[Defect] = []
    # Cuff/opening suppression: ignore a very large "hole" near the bottom band
    # (often just the glove opening showing background).
    ys = np.where(mf > 0)[0]
    if ys.size:
        y0, y1 = int(ys.min()), int(ys.max())
        gh = max(1, y1 - y0 + 1)
        cuff_y = y1 - int(round(0.18 * gh))
    else:
        cuff_y = int(mf.shape[0] * 0.85)

    for i in range(1, num):
        x, y, w, h, area = [int(v) for v in stats[i].tolist()]
        if area < 40:
            continue
        bbox = BoundingBox(x=x, y=y, w=w, h=h)
        aspect = (max(w, h) / (min(w, h) + 1e-6))
        area_norm = float(area) / glove_area

        cy = y + (h / 2.0)
        if cy >= float(cuff_y) and area_norm > 0.02:
            # Likely cuff opening rather than a puncture/hole defect.
            continue

        comp = (labels == i).astype(np.uint8)
        hole_med = np.median(lab[comp > 0], axis=0) if comp.sum() else glove_med
        d_bg = float(np.linalg.norm(hole_med - bg_med))
        d_glove = float(np.linalg.norm(hole_med - glove_med))
        l_delta = float(glove_med[0] - hole_med[0])

        # Reject color anomalies that are not background-like (common false "hole" on stains/discoloration).
        if not (d_bg + 3.0 < d_glove and l_delta > 8.0):
            if area_norm > 0.003:
                continue

        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            peri = float(cv2.arcLength(c, True)) + 1e-6
            circ = float((4.0 * np.pi * float(area)) / (peri * peri))
        else:
            circ = 0.0

        # Confidence heuristic: bigger voids + high circularity are strong evidence.
        size_score = min(1.0, area_norm / 0.008)  # 0.8% of glove area → max confidence
        # Elongated holes are more "tear"-like; rounder are "hole".
        if aspect >= 2.2:
            elong = min(1.0, (float(aspect) - 2.2) / 2.5)
            score = 0.60 + 0.40 * (0.6 * size_score + 0.4 * elong)
            out.append(
                Defect(
                    label="tear",
                    score=_clamp01(score),
                    bbox=bbox,
                    meta={
                        "area": area,
                        "aspect": float(aspect),
                        "area_norm": area_norm,
                        "d_bg": d_bg,
                        "d_glove": d_glove,
                    },
                )
            )
        else:
            score = 0.60 + 0.40 * (0.55 * size_score + 0.45 * min(1.0, max(0.0, circ)))
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
                    },
                )
            )
    return out


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
    if density < 0.10:
        return out

    # Detect strong long lines for "damaged by fold"
    gray = cv2.cvtColor(ctx.bgr, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(gray, 60, 160)
    e[interior == 0] = 0
    h, w = mask01.shape[:2]
    max_side = float(max(h, w))

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
        hist, bin_edges = np.histogram(angles, bins=12, range=(0.0, 180.0), weights=lengths)
        best_bin = int(np.argmax(hist))
        a0 = float(bin_edges[best_bin])
        a1 = float(bin_edges[best_bin + 1])
        aligned = (angles >= a0) & (angles < a1)
        aligned_len = float(lengths[aligned].sum())
        if max_len > 0.40 * max_side and aligned_len > 0.65 * max_side:
            out.append(
                Defect(
                    label="damaged_by_fold",
                    score=_clamp01(min(0.97, 0.58 + 0.85 * float(density))),
                    bbox=None,
                    meta={
                        "edge_density": density,
                        "max_line_len": max_len,
                        "aligned_len": aligned_len,
                        "method": "lsd",
                    },
                )
            )
            return out

    # Fallback to Hough-based long-line criterion.
    lines = cv2.HoughLinesP(e, 1, np.pi / 180, threshold=70, minLineLength=70, maxLineGap=10)
    if lines is not None and len(lines) > 0:
        lengths = []
        for (x1, y1, x2, y2) in lines.reshape(-1, 4):
            lengths.append(float(np.hypot(x2 - x1, y2 - y1)))
        max_len = max(lengths) if lengths else 0.0
        if max_len > 0.42 * max_side:
            out.append(
                Defect(
                    label="damaged_by_fold",
                    score=_clamp01(min(0.95, 0.55 + 0.9 * float(density))),
                    bbox=None,
                    meta={"edge_density": density, "max_line_len": max_len, "method": "hough"},
                )
            )
            return out

    # Otherwise, treat as wrinkles/dent (global texture/edge anomalies).
    out.append(
        Defect(
            label="wrinkles_dent",
            score=_clamp01(min(0.90, 0.45 + 0.9 * float(density))),
            bbox=None,
            meta={"edge_density": density},
        )
    )
    return out


def _spot_stain_discoloration(ctx: DefectDetectionContext) -> list[Defect]:
    mask01 = (ctx.glove_mask > 0).astype(np.uint8)
    # Suppress blob defects if segmentation seems to include too much background.
    glove_area = float(mask01.sum())
    h, w = mask01.shape[:2]
    if glove_area < 0.05 * h * w:
        return []
    x, y, ww, hh = cv2.boundingRect(mask01)
    box_area = float(ww * hh) + 1e-6
    extent = glove_area / box_area
    # If extent is very low, the mask is sparse inside a big box (often background leakage).
    if extent < 0.22:
        return []

    cand01, labels, stats = candidate_blobs(ctx.anomaly.combined, ctx.glove_mask)
    if stats.shape[0] == 0:
        return []

    lab = cv2.cvtColor(ctx.bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    vals = lab[mask01 > 0]
    if vals.size == 0:
        return []
    med = np.median(vals, axis=0)

    out: list[Defect] = []
    small_spots = 0
    glove_area = float(mask01.sum()) + 1e-6
    for i in range(1, stats.shape[0]):
        x, y, w, h, area = [int(v) for v in stats[i].tolist()]
        if area < 25:
            continue
        # Skip huge blobs (usually background segmentation imperfections).
        if area > 0.25 * glove_area:
            continue
        blob = (labels == i).astype(np.uint8)
        mean_de = float(ctx.anomaly.color[blob > 0].mean()) if blob.sum() else 0.0
        if mean_de < 0.35:
            continue
        bbox = BoundingBox(x=x, y=y, w=w, h=h)

        mean_L = float(lab[:, :, 0][blob > 0].mean()) if blob.sum() else float(med[0])
        delta_L = (mean_L - float(med[0])) / 255.0
        # Chroma anomaly (ignore brightness; more robust to shadows).
        ab = lab[:, :, 1:3]
        med_ab = med[1:3].reshape(1, 1, 2)
        de_ab = np.linalg.norm(ab - med_ab, axis=2)
        mean_de_ab = float(de_ab[blob > 0].mean()) if blob.sum() else 0.0

        # Heuristic label based on size and brightness change.
        if area <= 180 and max(w, h) <= 30:
            small_spots += 1
            # Plastic contamination tends to be specular/bright and tiny.
            if delta_L > 0.10:
                out.append(Defect(label="plastic_contamination", score=_clamp01(0.55 + 0.5 * mean_de), bbox=bbox))
            continue

        if area >= 0.015 * glove_area:
            # Large area: discoloration vs stain/dirty via brightness direction.
            if delta_L < -0.06:
                out.append(Defect(label="stain_dirty", score=_clamp01(0.50 + 0.5 * mean_de), bbox=bbox))
            else:
                # Prefer discoloration when chroma differs (not just bright/dim).
                chroma_boost = min(0.25, mean_de_ab / 45.0)
                out.append(Defect(label="discoloration", score=_clamp01(0.55 + 0.45 * mean_de + chroma_boost), bbox=bbox))
        else:
            # Medium blobs are usually stains/dirty spots.
            out.append(Defect(label="stain_dirty", score=_clamp01(0.50 + 0.5 * mean_de), bbox=bbox))

    if small_spots >= 6:
        out.append(
            Defect(
                label="spotting",
                score=_clamp01(min(0.95, 0.35 + 0.08 * small_spots)),
                bbox=None,
                meta={"count": small_spots},
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
    thr = float(np.percentile(de_ab[core > 0], 97) + 3.0)
    thr = max(4.0, min(24.0, thr))
    cand = ((de_ab >= thr) & (mask01 > 0)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    dt = cv2.distanceTransform((mask01 * 255).astype(np.uint8), cv2.DIST_L2, 5)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    out: list[Defect] = []
    for i in range(1, num):
        x, y, ww, hh, area = [int(v) for v in stats[i].tolist()]
        if area < 55:
            continue
        area_frac = float(area) / glove_area
        if area_frac < 0.0040 or area_frac > 0.12:
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
        delta_L = abs(mean_L - float(med[0])) / 255.0
        # Strong discoloration should have chroma shift, not just brightness shift.
        score = 0.45 + 0.45 * min(1.0, mean_de_ab / (thr + 8.0)) + 0.20 * max(0.0, 0.35 - delta_L)
        bbox = BoundingBox(x=x, y=y, w=ww, h=hh)
        out.append(Defect(label="discoloration", score=_clamp01(score), bbox=bbox))

    out.sort(key=lambda d: float(d.score), reverse=True)
    return out[:4]


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
    cuff = _cuff_region_mask(ctx.glove_mask_filled)
    if cuff.sum() < 150:
        return []

    mask01f = (ctx.glove_mask_filled > 0).astype(np.uint8)
    # Boundary band: dilated minus eroded gives a thickness-normalized boundary indicator.
    dil = cv2.dilate(mask01f, np.ones((7, 7), np.uint8), iterations=1)
    ero = cv2.erode(mask01f, np.ones((7, 7), np.uint8), iterations=1)
    boundary = ((dil > 0) & (ero == 0)).astype(np.uint8)
    boundary_cuff = (boundary & cuff).astype(np.uint8)

    # Boundary complexity: edge density near cuff boundary.
    gray = cv2.cvtColor(ctx.bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edges = ((edges > 0).astype(np.uint8) & boundary_cuff).astype(np.uint8)
    complexity = float(edges.sum()) / float(boundary_cuff.sum() + 1e-6)

    # Thickness: distance transform inside cuff.
    dt = cv2.distanceTransform((mask01f * 255).astype(np.uint8), cv2.DIST_L2, 5)
    thickness = float(np.median(dt[cuff > 0])) if cuff.sum() else 0.0
    thickness_all = float(np.median(dt[mask01f > 0])) if mask01f.sum() else 1.0
    rel_thickness = thickness / (thickness_all + 1e-6)

    out: list[Defect] = []
    if complexity > 0.18:
        out.append(
            Defect(
                label="improper_roll",
                score=_clamp01(min(0.95, 0.35 + 1.8 * complexity)),
                bbox=None,
                meta={"cuff_complexity": complexity},
            )
        )
    if rel_thickness < 0.75:
        out.append(
            Defect(
                label="incomplete_beading",
                score=_clamp01(min(0.95, 0.35 + 0.8 * (0.75 - rel_thickness) / 0.75)),
                bbox=None,
                meta={"rel_thickness": rel_thickness},
            )
        )
    return out


def _finger_count_anomaly(ctx: DefectDetectionContext) -> list[Defect]:
    """
    Detect missing/extra fingers using silhouette-only structural features.
    We estimate a finger count from the glove mask, and flag anomalies vs 5.
    """
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
            gap_evidence = bool(gap_ratio >= 1.45 and gap_w >= max(16, int(round(rot_mask.shape[1] * 0.06))))

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
    single_profile_support = bool(single_profile_missing and (short_tip_supported or gap_evidence))
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
        }
    )

    if have_both:
        # If methods disagree too much, avoid weak +/-1 decisions.
        if abs(p_count - h_count) >= 2 and abs(diff) <= 1 and not profile_hull_conflict_support:
            return []

    if diff < 0:
        # For a single-finger deficit, require stronger structural evidence.
        # This avoids false positives when adjacent fingers are close and peaks merge.
        if abs(diff) == 1:
            if have_both:
                if not (strong_missing_count or truncated_tip_evidence):
                    return []
                # Allow small (1-count) estimator disagreement when both methods still
                # indicate a deficit. This is common for cut fingertips where one method
                # undercounts slightly more than the other.
                if abs(p_count - h_count) > 1 and not truncated_tip_evidence:
                    return []
                if not (gap_evidence or short_tip_supported or count_consensus_support or count_deficit_support):
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
            score += 0.14
        if count_deficit_support:
            score += 0.08
        if gap_evidence:
            score += 0.10
        if short_tip_supported:
            score += 0.12
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
            # This specific conflict pattern often corresponds to a fingertip puncture/cut:
            # shape remains mostly 5-finger, but one tip is structurally shortened.
            label = "hole" if (profile_hull_conflict_support and not short_tip_supported) else "missing_finger"
            out.append(Defect(label=label, score=_clamp01(score), bbox=bbox_orig, meta=meta))
    elif diff > 0:
        if abs(diff) == 1 and not gap_evidence:
            return []
        score = 0.55 + 0.16 * min(3, abs(diff))
        if have_both and p_count >= 6 and h_count >= 6:
            score += 0.08
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

    gray = cv2.cvtColor(ctx.bgr, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(gray, 60, 160)
    e = ((e > 0).astype(np.uint8) & boundary).astype(np.uint8)
    edge_density = float(e.sum()) / float(boundary.sum() + 1e-6)

    # Conservative threshold; tune per dataset.
    if edge_density > 0.22:
        return [
            Defect(
                label="inside_out",
                score=_clamp01(min(0.95, 0.25 + 2.5 * edge_density)),
                bbox=None,
                meta={"boundary_edge_density": edge_density},
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
    focus_only: bool = False,
    allowed_labels: set[str] | None = None,
) -> tuple[list[Defect], AnomalyMaps]:
    anomaly = build_anomaly_maps(bgr, glove_mask)
    ctx = DefectDetectionContext(
        bgr=bgr,
        glove_mask=glove_mask,
        glove_mask_filled=glove_mask_filled,
        anomaly=anomaly,
    )

    defects: list[Defect] = []
    if focus_only:
        defects.extend([d for d in _detect_holes(bgr, glove_mask, glove_mask_filled) if d.label == "hole"])
        defects.extend(_discoloration_only(ctx))
        defects.extend([d for d in _edge_fold_wrinkle(ctx) if d.label == "damaged_by_fold"])
        defects.extend(_finger_count_anomaly(ctx))
    else:
        defects.extend(_detect_holes(bgr, glove_mask, glove_mask_filled))
        defects.extend(_spot_stain_discoloration(ctx))
        defects.extend(_edge_fold_wrinkle(ctx))
        defects.extend(_roll_and_beading(ctx))
        defects.extend(_finger_count_anomaly(ctx))
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
    return out, anomaly
