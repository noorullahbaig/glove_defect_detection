from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


@dataclass(frozen=True)
class Defect:
    label: str
    score: float
    bbox: BoundingBox | None = None
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class InferenceResult:
    glove_mask: np.ndarray  # uint8 (0/255), shape (H,W)
    glove_type: str
    glove_type_score: float
    defects: list[Defect]

