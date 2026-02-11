from __future__ import annotations

import numpy as np


def lbp8(gray: np.ndarray) -> np.ndarray:
    """
    Basic 8-neighbor LBP (radius=1) implementation.

    Returns uint8 codes in [0,255]. This is intentionally simple to avoid extra deps.
    """
    if gray.ndim != 2:
        raise ValueError("lbp8 expects a 2D grayscale array")
    g = gray.astype(np.int16)
    h, w = g.shape
    out = np.zeros((h, w), dtype=np.uint8)

    center = g[1:-1, 1:-1]
    neighbors = [
        g[0:-2, 0:-2],
        g[0:-2, 1:-1],
        g[0:-2, 2:],
        g[1:-1, 2:],
        g[2:, 2:],
        g[2:, 1:-1],
        g[2:, 0:-2],
        g[1:-1, 0:-2],
    ]
    for i, n in enumerate(neighbors):
        out[1:-1, 1:-1] |= ((n >= center) << i).astype(np.uint8)
    return out


def lbp_hist(gray: np.ndarray, mask01: np.ndarray | None = None, bins: int = 256) -> np.ndarray:
    codes = lbp8(gray)
    if mask01 is None:
        vals = codes.ravel()
    else:
        vals = codes[mask01 > 0].ravel()
    hist = np.bincount(vals, minlength=bins).astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist

