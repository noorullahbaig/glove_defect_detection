from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .lbp import lbp8


@dataclass(frozen=True)
class AnomalyMaps:
    color: np.ndarray  # float32 [0,1]
    texture: np.ndarray  # float32 [0,1]
    edges: np.ndarray  # float32 [0,1]
    combined: np.ndarray  # float32 [0,1]


def _normalize01(x: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    vals = x[mask01 > 0]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    hi = float(np.percentile(vals, 95))
    if hi <= 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    out = np.clip(x / hi, 0.0, 1.0).astype(np.float32)
    out[mask01 == 0] = 0.0
    return out


def color_anomaly_map(bgr: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    vals = lab[mask01 > 0]
    if vals.size == 0:
        return np.zeros(mask01.shape, dtype=np.float32)
    med = np.median(vals, axis=0)
    # Approximate DeltaE with Euclidean distance in LAB.
    de = np.linalg.norm(lab - med.reshape(1, 1, 3), axis=2).astype(np.float32)
    return _normalize01(de, mask01)


def texture_anomaly_map(bgr: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    codes = lbp8(gray)
    vals = codes[mask01 > 0].ravel()
    if vals.size == 0:
        return np.zeros(mask01.shape, dtype=np.float32)
    hist = np.bincount(vals, minlength=256).astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    p = hist[codes]
    # Rare LBP patterns become high anomaly. Clip to keep stable.
    anom = np.clip(1.0 - p / (np.max(hist) + 1e-6), 0.0, 1.0).astype(np.float32)
    anom[mask01 == 0] = 0.0
    return anom


def edge_map(bgr: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160).astype(np.float32) / 255.0
    edges[mask01 == 0] = 0.0
    # Local edge density (texture/crease indicator)
    k = np.ones((11, 11), np.float32) / 121.0
    density = cv2.filter2D(edges, -1, k)
    return _normalize01(density, mask01)


def build_anomaly_maps(bgr: np.ndarray, glove_mask: np.ndarray) -> AnomalyMaps:
    mask01 = (glove_mask > 0).astype(np.uint8)
    c = color_anomaly_map(bgr, mask01)
    t = texture_anomaly_map(bgr, mask01)
    e = edge_map(bgr, mask01)
    combined = np.clip(0.55 * c + 0.25 * t + 0.20 * e, 0.0, 1.0).astype(np.float32)
    combined[mask01 == 0] = 0.0
    return AnomalyMaps(color=c, texture=t, edges=e, combined=combined)


def candidate_blobs(combined: np.ndarray, glove_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Threshold the combined anomaly map within the glove region and return:
    - candidate_mask01
    - labels image
    - stats array (as in connectedComponentsWithStats)
    """
    mask01 = (glove_mask > 0).astype(np.uint8)
    vals = combined[mask01 > 0]
    if vals.size == 0:
        z = np.zeros(mask01.shape, np.uint8)
        return z, z, np.zeros((0, 5), np.int32)

    # Otsu on scaled values within mask.
    scaled = (combined * 255).astype(np.uint8)
    inside = scaled[mask01 > 0]
    th_val, _th_img = cv2.threshold(inside, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Make the threshold a bit more conservative to reduce background leakage.
    th_val = min(255, int(th_val) + 10)
    cand01 = ((scaled >= int(th_val)) & (mask01 > 0)).astype(np.uint8)

    cand01 = cv2.morphologyEx(cand01, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    cand01 = cv2.morphologyEx(cand01, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(cand01, connectivity=8)
    if num <= 1:
        return cand01, labels.astype(np.int32), stats
    return cand01, labels.astype(np.int32), stats
