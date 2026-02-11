from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .labels import DEFECT_LABELS, GLOVE_TYPES


@dataclass(frozen=True)
class DatasetRow:
    file: str
    glove_type: str
    defect_labels: list[str]
    split: str
    lighting: str
    background: str
    notes: str


def load_labels_csv(path: str | Path = "data/labels.csv") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing labels file: {path}")
    df = pd.read_csv(path)
    required = ["file", "glove_type", "defect_labels", "split"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"labels.csv missing columns: {missing}")
    return df


def parse_defect_labels(s: str) -> list[str]:
    s = (s or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split("|") if p.strip()]
    # Keep only known labels (helps when students accidentally type variations).
    return [p for p in parts if p in DEFECT_LABELS]


def validate_labels_df(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if "glove_type" in df.columns:
        bad = sorted(set(df["glove_type"].dropna().astype(str)) - set(GLOVE_TYPES))
        if bad:
            errors.append(f"Unknown glove_type values: {bad}")
    if "split" in df.columns:
        bad = sorted(set(df["split"].dropna().astype(str)) - {"train", "val", "test"})
        if bad:
            errors.append(f"Unknown split values: {bad}")
    return errors

