from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from .labels import GLOVE_TYPES


@dataclass
class GloveTypeModel:
    encoder: LabelEncoder
    clf: RandomForestClassifier

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        probs = self.clf.predict_proba(features.reshape(1, -1))[0]
        idx = int(np.argmax(probs))
        return str(self.encoder.inverse_transform([idx])[0]), float(probs[idx])


def train_glove_type_model(x: np.ndarray, y: list[str]) -> GloveTypeModel:
    encoder = LabelEncoder()
    encoder.fit(GLOVE_TYPES)
    y_enc = encoder.transform(y)
    clf = RandomForestClassifier(
        n_estimators=350,
        random_state=42,
        max_depth=None,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    clf.fit(x, y_enc)
    return GloveTypeModel(encoder=encoder, clf=clf)


def save_glove_type_model(model: GloveTypeModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"encoder": model.encoder, "clf": model.clf}, path)


def load_glove_type_model(path: str | Path) -> GloveTypeModel:
    obj = joblib.load(path)
    return GloveTypeModel(encoder=obj["encoder"], clf=obj["clf"])

