from __future__ import annotations

import cv2
import numpy as np

from .types import Defect


def overlay_mask(bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int] = (0, 255, 0), alpha: float = 0.35) -> np.ndarray:
    out = bgr.copy()
    m = (mask > 0)
    if not m.any():
        return out
    overlay = out.copy()
    overlay[m] = (np.array(color_bgr, dtype=np.uint8)).reshape(1, 1, 3)
    return cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0.0)


def draw_defects(bgr: np.ndarray, defects: list[Defect]) -> np.ndarray:
    out = bgr.copy()
    for d in defects:
        if d.bbox is None:
            continue
        x1, y1, x2, y2 = d.bbox.as_xyxy()
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            out,
            f"{d.label} ({d.score:.2f})",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return out

