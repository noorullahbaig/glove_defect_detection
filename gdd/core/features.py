from __future__ import annotations

import cv2
import numpy as np

from .lbp import lbp_hist


def color_hist_lab(bgr: np.ndarray, mask01: np.ndarray, bins_per_channel: int = 16) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hist = []
    for ch in range(3):
        h = cv2.calcHist([lab], [ch], mask01.astype(np.uint8), [bins_per_channel], [0, 256])
        h = h.astype(np.float32).ravel()
        hist.append(h)
    vec = np.concatenate(hist, axis=0)
    vec /= (vec.sum() + 1e-6)
    return vec


def shape_features(mask01_filled: np.ndarray) -> np.ndarray:
    """
    Shape features from the filled glove silhouette.
    """
    cnts, _ = cv2.findContours(mask01_filled.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        # 6 scalar features + 7 Hu moments = 13 dims (must be stable length)
        return np.zeros((13,), dtype=np.float32)
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    x, y, w, h = cv2.boundingRect(c)
    rect_area = float(w * h) + 1e-6
    extent = area / rect_area
    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull)) + 1e-6
    solidity = area / hull_area
    peri = float(cv2.arcLength(c, True)) + 1e-6
    compactness = (4.0 * np.pi * area) / (peri * peri)

    moments = cv2.moments(c)
    hu = cv2.HuMoments(moments).astype(np.float32).ravel()
    # Log-scale Hu moments for numerical stability.
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    feat = np.array([area, w, h, extent, solidity, compactness], dtype=np.float32)
    return np.concatenate([feat, hu], axis=0)


def glove_type_features(bgr: np.ndarray, glove_mask: np.ndarray, glove_mask_filled: np.ndarray) -> np.ndarray:
    """
    Feature vector for glove type classification (nitrile/plastic/fabric).
    """
    mask01 = (glove_mask > 0).astype(np.uint8)
    mask01f = (glove_mask_filled > 0).astype(np.uint8)
    ch = color_hist_lab(bgr, mask01, bins_per_channel=16)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    th = lbp_hist(gray, mask01, bins=256)
    sh = shape_features(mask01f)
    return np.concatenate([ch, th, sh], axis=0).astype(np.float32)
