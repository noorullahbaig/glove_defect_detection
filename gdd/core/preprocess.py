from __future__ import annotations

import cv2
import numpy as np


def normalize_illumination(bgr: np.ndarray) -> np.ndarray:
    """
    Normalize illumination using CLAHE on the L channel in LAB space.

    This makes downstream thresholding and feature extraction less sensitive to
    lighting changes (bright daylight vs warm/dim indoor).
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def denoise(bgr: np.ndarray) -> np.ndarray:
    """
    Lightweight denoising that preserves edges reasonably well.
    """
    # Median is robust to salt/pepper noise; bilateral helps retain edges.
    bgr2 = cv2.medianBlur(bgr, 3)
    return cv2.bilateralFilter(bgr2, d=7, sigmaColor=50, sigmaSpace=50)


def preprocess(bgr: np.ndarray) -> np.ndarray:
    return denoise(normalize_illumination(bgr))

