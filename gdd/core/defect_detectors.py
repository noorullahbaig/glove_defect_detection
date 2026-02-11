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
    for i in range(1, num):
        x, y, w, h, area = [int(v) for v in stats[i].tolist()]
        if area < 40:
            continue
        bbox = BoundingBox(x=x, y=y, w=w, h=h)
        aspect = (max(w, h) / (min(w, h) + 1e-6))
        area_norm = float(area) / glove_area

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
    mask01 = (ctx.glove_mask > 0).astype(np.uint8)
    edges = (ctx.anomaly.edges * 255).astype(np.uint8)
    edges[mask01 == 0] = 0

    # Global edge density (in a smoothed sense) indicates wrinkles/texture anomalies.
    density = float(ctx.anomaly.edges[mask01 > 0].mean()) if mask01.sum() else 0.0
    out: list[Defect] = []
    if density < 0.08:
        return out

    # Detect strong long lines for "damaged by fold"
    e = cv2.Canny(cv2.cvtColor(ctx.bgr, cv2.COLOR_BGR2GRAY), 60, 160)
    e[mask01 == 0] = 0
    lines = cv2.HoughLinesP(e, 1, np.pi / 180, threshold=80, minLineLength=60, maxLineGap=12)
    if lines is not None and len(lines) > 0:
        h, w = mask01.shape[:2]
        lengths = []
        for (x1, y1, x2, y2) in lines.reshape(-1, 4):
            lengths.append(float(np.hypot(x2 - x1, y2 - y1)))
        max_len = max(lengths) if lengths else 0.0
        if max_len > 0.35 * max(h, w):
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
                out.append(Defect(label="discoloration", score=_clamp01(0.50 + 0.5 * mean_de), bbox=bbox))
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


def _missing_finger(ctx: DefectDetectionContext) -> list[Defect]:
    """
    Approximate missing-finger detection using a simple silhouette profile peak count.

    Works best when:
    - the glove is roughly upright in frame
    - fingers are separated or at least distinguishable in the silhouette
    """
    mask = (ctx.glove_mask_filled > 0).astype(np.uint8)
    if mask.sum() < 300:
        return []

    ys = np.where(mask > 0)[0]
    y0, y1 = int(ys.min()), int(ys.max())
    h = y1 - y0 + 1
    top = mask[y0 : y0 + int(round(0.45 * h)), :]
    if top.sum() < 200:
        return []

    # Column-wise profile in top region.
    prof = top.sum(axis=0).astype(np.float32)
    if prof.max() <= 1:
        return []
    prof /= (prof.max() + 1e-6)

    # Smooth with a small window.
    win = 9
    kernel = np.ones((win,), np.float32) / float(win)
    prof_s = np.convolve(prof, kernel, mode="same")

    # Peak count above a threshold with non-max suppression.
    thresh = 0.32
    peaks: list[int] = []
    for i in range(2, len(prof_s) - 2):
        if prof_s[i] < thresh:
            continue
        if prof_s[i] >= prof_s[i - 1] and prof_s[i] >= prof_s[i + 1] and prof_s[i] >= prof_s[i - 2] and prof_s[i] >= prof_s[i + 2]:
            if not peaks or (i - peaks[-1]) > 18:
                peaks.append(i)

    # Typical hand gloves have 5 fingers; allow 4–6 peaks due to thumb merging etc.
    if len(peaks) <= 3:
        return [Defect(label="missing_finger", score=_clamp01(0.78), bbox=None, meta={"peaks": len(peaks)})]
    if len(peaks) == 4:
        return [Defect(label="missing_finger", score=_clamp01(0.55), bbox=None, meta={"peaks": len(peaks)})]
    return []


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
    defects.extend(_missing_finger(ctx))
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
