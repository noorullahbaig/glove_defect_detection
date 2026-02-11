from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class LoadedImage:
    bgr: np.ndarray  # uint8 HxWx3
    path: str


def read_image(path: str | Path) -> LoadedImage:
    path = str(path)
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return LoadedImage(bgr=bgr, path=path)


def resize_max_side(bgr: np.ndarray, max_side: int = 900) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = max(h, w) / float(max_side)
    if scale <= 1.0:
        return bgr
    new_w = int(round(w / scale))
    new_h = int(round(h / scale))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

