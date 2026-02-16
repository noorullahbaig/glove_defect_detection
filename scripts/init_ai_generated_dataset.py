from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gdd.core.labels import DEFECT_LABELS, GLOVE_TYPES


def _iter_image_files(root: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return sorted(out)


def _assign_splits(keys: list[str], seed: int, train: float, val: float) -> dict[str, str]:
    """
    Assign split labels per group key using a stable shuffle.
    We ensure tiny groups don't end up with empty train by keeping at least 1 in train.
    """
    rng = random.Random(int(seed))
    idxs = list(range(len(keys)))
    rng.shuffle(idxs)

    n = len(keys)
    if n <= 0:
        return {}
    n_train = max(1, int(round(train * n)))
    n_val = int(round(val * n))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    split_by_idx: dict[int, str] = {}
    for j, i in enumerate(idxs):
        if j < n_train:
            split_by_idx[i] = "train"
        elif j < n_train + n_val:
            split_by_idx[i] = "val"
        else:
            split_by_idx[i] = "test"
    return {keys[i]: split_by_idx[i] for i in range(n)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ai_generated", help="Dataset root (contains images/ and labels.csv)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    args = ap.parse_args()

    root = Path(args.root)
    img_root = root / "images"
    root.mkdir(parents=True, exist_ok=True)
    img_root.mkdir(parents=True, exist_ok=True)

    # Create recommended folders (structure only).
    for gt in GLOVE_TYPES:
        (img_root / gt / "normal").mkdir(parents=True, exist_ok=True)
        for lab in DEFECT_LABELS:
            (img_root / gt / lab).mkdir(parents=True, exist_ok=True)

    paths = _iter_image_files(img_root)
    labels_path = root / "labels.csv"
    if not paths:
        # Create an empty labels file with the expected header.
        if not labels_path.exists():
            pd.DataFrame(columns=["file", "glove_type", "defect_labels", "split", "lighting", "background", "notes"]).to_csv(labels_path, index=False)
        print(f"No images found under {img_root}. Wrote template: {labels_path}")
        return

    rows: list[dict] = []
    for p in paths:
        # Expect: images/<glove_type>/<defect_label>/<file>
        try:
            rel = p.relative_to(img_root)
        except Exception:
            continue
        parts = rel.parts
        if len(parts) < 3:
            continue
        glove_type = str(parts[0])
        defect_folder = str(parts[1])
        if glove_type not in set(GLOVE_TYPES):
            continue
        defect = "" if defect_folder == "normal" else defect_folder
        if defect and defect not in set(DEFECT_LABELS):
            continue

        rows.append(
            {
                "file": str(p.as_posix()),
                "glove_type": glove_type,
                "defect_labels": defect,
                "split": "",  # filled later
                "lighting": "ai_generated",
                "background": "ai_generated",
                "notes": f"source=ai path={rel.as_posix()}",
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        pd.DataFrame(columns=["file", "glove_type", "defect_labels", "split", "lighting", "background", "notes"]).to_csv(labels_path, index=False)
        print(f"No valid images found under {img_root}. Wrote template: {labels_path}")
        return

    # Stratify by glove type + defect label string.
    df["strata"] = df["glove_type"].astype(str) + "|" + df["defect_labels"].astype(str)
    split_map: dict[str, str] = {}
    for strata, g in df.groupby("strata"):
        keys = g["file"].astype(str).tolist()
        split_map.update(_assign_splits(keys, seed=int(args.seed), train=float(args.train), val=float(args.val)))
    df["split"] = df["file"].map(split_map).fillna("train")
    df = df.drop(columns=["strata"])

    df.to_csv(labels_path, index=False)
    counts = df["split"].value_counts().to_dict()
    print(f"Wrote: {labels_path} (rows={len(df)} splits={counts})")


if __name__ == "__main__":
    main()

