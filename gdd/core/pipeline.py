from __future__ import annotations

from pathlib import Path

import numpy as np

from .defect_detectors import detect_defects
from .features import glove_type_features
from .glove_type_model import GloveTypeModel, load_glove_type_model
from .preprocess import preprocess
from .segmentation import segment_glove
from .types import InferenceResult


DEFAULT_GLOVE_TYPE_MODEL_PATH = Path("gdd/models/glove_type.joblib")


class GDDPipeline:
    def __init__(self, glove_type_model: GloveTypeModel | None = None):
        self.glove_type_model = glove_type_model

    @classmethod
    def load_default(cls) -> "GDDPipeline":
        model = None
        if DEFAULT_GLOVE_TYPE_MODEL_PATH.exists():
            try:
                model = load_glove_type_model(DEFAULT_GLOVE_TYPE_MODEL_PATH)
            except Exception:
                # Model may be stale (wrong classes or wrong feature length). Treat as missing.
                model = None
        return cls(glove_type_model=model)

    def infer(
        self,
        bgr: np.ndarray,
        focus_only: bool = False,
        allowed_labels: set[str] | None = None,
    ) -> InferenceResult:
        bgr_p = preprocess(bgr)
        seg = segment_glove(bgr_p)
        defects, _anom = detect_defects(
            bgr_p,
            seg.glove_mask,
            seg.glove_mask_filled,
            focus_only=bool(focus_only),
            allowed_labels=allowed_labels,
        )

        glove_type = "unknown"
        glove_type_score = 0.0
        if self.glove_type_model is not None:
            feats = glove_type_features(bgr_p, seg.glove_mask, seg.glove_mask_filled)
            try:
                glove_type, glove_type_score = self.glove_type_model.predict(feats)
            except Exception:
                glove_type, glove_type_score = "unknown", 0.0

        return InferenceResult(
            glove_mask=seg.glove_mask,
            glove_type=glove_type,
            glove_type_score=float(glove_type_score),
            defects=defects,
        )
