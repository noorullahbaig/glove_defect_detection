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

def _rotate_mask_to_upright(mask01: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Rotate the binary mask so the main axis of the glove is roughly vertical.
    Returns (rotated_mask01, angle_degrees_applied).
    """
    pts = np.column_stack(np.where(mask01 > 0))  # (y,x)
    if pts.shape[0] < 200:
        return mask01, 0.0

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
    return (out > 0).astype(np.uint8), float(rot)


def _count_profile_peaks(mask01: np.ndarray) -> int:
    """
    Count finger-like peaks by projecting the top portion of the mask onto the x-axis.
    This is tolerant to lighting changes since it uses only silhouette geometry.
    """
    if mask01.sum() < 300:
        return 0

    ys = np.where(mask01 > 0)[0]
    if ys.size == 0:
        return 0
    y0, y1 = int(ys.min()), int(ys.max())
    h = y1 - y0 + 1
    top = mask01[y0 : y0 + int(round(0.50 * h)), :]
    if top.sum() < 200:
        return 0

    prof = top.sum(axis=0).astype(np.float32)
    if prof.max() <= 1:
        return 0
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
    return int(len(peaks))


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
    rot_mask, rot = _rotate_mask_to_upright(mask)
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


def _detect_holes(glove_mask: np.ndarray, glove_mask_filled: np.ndarray) -> list[Defect]:
    """
    Detect holes/tears by finding regions that are inside the filled silhouette but missing in the raw mask.
    """
    m = (glove_mask > 0).astype(np.uint8)
    mf = (glove_mask_filled > 0).astype(np.uint8)
    glove_area = float(mf.sum()) + 1e-6
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
                    meta={"area": area, "aspect": float(aspect), "area_norm": area_norm},
                )
            )
        else:
            score = 0.60 + 0.40 * (0.55 * size_score + 0.45 * min(1.0, max(0.0, circ)))
            out.append(
                Defect(
                    label="hole",
                    score=_clamp01(score),
                    bbox=bbox,
                    meta={"area": area, "aspect": float(aspect), "circularity": circ, "area_norm": area_norm},
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
    e = cv2.Canny(cv2.cvtColor(ctx.bgr, cv2.COLOR_BGR2GRAY), 60, 160)
    e[interior == 0] = 0
    lines = cv2.HoughLinesP(e, 1, np.pi / 180, threshold=70, minLineLength=70, maxLineGap=10)
    if lines is not None and len(lines) > 0:
        h, w = mask01.shape[:2]
        lengths = []
        for (x1, y1, x2, y2) in lines.reshape(-1, 4):
            lengths.append(float(np.hypot(x2 - x1, y2 - y1)))
        max_len = max(lengths) if lengths else 0.0
        # Strong fold: a dominant long straight line across a big fraction of the glove.
        if max_len > 0.42 * max(h, w):
            out.append(
                Defect(
                    label="damaged_by_fold",
                    score=_clamp01(min(0.95, 0.55 + 0.9 * float(density))),
                    bbox=None,
                    meta={"edge_density": density, "max_line_len": max_len},
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
    count, meta = _estimate_finger_count(ctx.glove_mask_filled)
    if count <= 0:
        return []

    expected = 5
    diff = int(count) - expected
    out: list[Defect] = []
    meta = dict(meta)
    meta.update({"finger_count": int(count), "expected": expected})

    if diff < 0:
        score = 0.55 + 0.18 * min(3, abs(diff))
        out.append(Defect(label="missing_finger", score=_clamp01(score), bbox=None, meta=meta))
    elif diff > 0:
        score = 0.55 + 0.16 * min(3, abs(diff))
        out.append(Defect(label="extra_fingers", score=_clamp01(score), bbox=None, meta=meta))
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


def detect_defects(bgr: np.ndarray, glove_mask: np.ndarray, glove_mask_filled: np.ndarray) -> tuple[list[Defect], AnomalyMaps]:
    anomaly = build_anomaly_maps(bgr, glove_mask)
    ctx = DefectDetectionContext(
        bgr=bgr,
        glove_mask=glove_mask,
        glove_mask_filled=glove_mask_filled,
        anomaly=anomaly,
    )

    defects: list[Defect] = []
    defects.extend(_detect_holes(glove_mask, glove_mask_filled))
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
    return out, anomaly
