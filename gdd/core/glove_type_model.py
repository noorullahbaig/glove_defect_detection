from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .labels import GLOVE_TYPES


@dataclass
class GloveTypeModel:
    encoder: LabelEncoder
    clf: object

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        x = features.reshape(1, -1)
        if hasattr(self.clf, "predict_proba"):
            probs = self.clf.predict_proba(x)[0]
            idx = int(np.argmax(probs))
            return str(self.encoder.inverse_transform([idx])[0]), float(probs[idx])
        pred = self.clf.predict(x)[0]
        return str(self.encoder.inverse_transform([int(pred)])[0]), 0.5


def train_glove_type_model(x: np.ndarray, y: list[str], model_type: str = "logreg") -> GloveTypeModel:
    encoder = LabelEncoder()
    encoder.fit(GLOVE_TYPES)
    y_enc = encoder.transform(y)
    model_type = str(model_type).lower().strip()

    if model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=450,
            random_state=42,
            max_depth=None,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        clf.fit(x, y_enc)
        return GloveTypeModel(encoder=encoder, clf=clf)

    if model_type == "logreg":
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2500,
                        n_jobs=-1,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        clf.fit(x, y_enc)
        return GloveTypeModel(encoder=encoder, clf=clf)

    raise ValueError(f"Unknown model_type: {model_type} (use 'logreg' or 'rf')")


def save_glove_type_model(model: GloveTypeModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"encoder": model.encoder, "clf": model.clf}, path)


def load_glove_type_model(path: str | Path) -> GloveTypeModel:
    obj = joblib.load(path)
    return GloveTypeModel(encoder=obj["encoder"], clf=obj["clf"])
