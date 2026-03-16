from __future__ import annotations

import argparse
import json
import random
from collections import Counter
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


def _assign_balanced_splits(keys: list[str], seed: int) -> dict[str, str]:
    """
    Rebalance tiny strata for evaluation:
    - n == 1: train
    - n in {2, 3}: 1 test, rest train
    - n >= 4: 1 test, 1 val, rest train
    """
    rng = random.Random(int(seed))
    ordered = list(keys)
    rng.shuffle(ordered)
    n = len(ordered)
    if n <= 0:
        return {}
    if n == 1:
        return {ordered[0]: "train"}
    if n in {2, 3}:
        out = {k: "train" for k in ordered}
        out[ordered[0]] = "test"
        return out
    out = {k: "train" for k in ordered}
    out[ordered[0]] = "test"
    out[ordered[1]] = "val"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ai_generated", help="Dataset root (contains images/ and labels.csv)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument(
        "--split-mode",
        default="balanced",
        choices=["balanced", "ratio"],
        help="How to assign splits within each (glove_type, defect) stratum.",
    )
    ap.add_argument(
        "--audit-json",
        default="",
        help="Optional path to write a manifest audit JSON.",
    )
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

    valid_glove_types = set(GLOVE_TYPES)
    valid_defects = set(DEFECT_LABELS)
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
        if glove_type not in valid_glove_types:
            continue
        defect = "" if defect_folder == "normal" else defect_folder
        if defect and defect not in valid_defects:
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

    # Enforce AI-generated folder as the only source of truth.
    root_prefix = img_root.as_posix().rstrip("/") + "/"
    df["file"] = df["file"].astype(str)
    df = df[df["file"].str.startswith(root_prefix)].copy()
    df = df.drop_duplicates(subset=["file"]).reset_index(drop=True)

    # Stratify by glove type + defect label string.
    df["strata"] = df["glove_type"].astype(str) + "|" + df["defect_labels"].astype(str)
    split_map: dict[str, str] = {}
    for strata, g in df.groupby("strata"):
        keys = g["file"].astype(str).tolist()
        if str(args.split_mode) == "balanced":
            split_map.update(_assign_balanced_splits(keys, seed=int(args.seed)))
        else:
            split_map.update(_assign_splits(keys, seed=int(args.seed), train=float(args.train), val=float(args.val)))
    df["split"] = df["file"].map(split_map).fillna("train")
    df = df.drop(columns=["strata"])

    df.to_csv(labels_path, index=False)
    counts = df["split"].value_counts().to_dict()
    print(f"Wrote: {labels_path} (rows={len(df)} splits={counts})")

    if str(args.audit_json).strip():
        defect_series = df["defect_labels"].fillna("").astype(str).replace({"": "normal"})
        strata_counts = Counter(zip(df["glove_type"].astype(str), defect_series.tolist()))
        split_by_strata: dict[str, dict[str, int]] = {}
        for (glove_type, defect), count in sorted(strata_counts.items()):
            sub = df[(df["glove_type"].astype(str) == glove_type) & (defect_series == defect)].copy()
            split_by_strata[f"{glove_type}|{defect}"] = {
                "total": int(count),
                "train": int((sub["split"].astype(str) == "train").sum()),
                "val": int((sub["split"].astype(str) == "val").sum()),
                "test": int((sub["split"].astype(str) == "test").sum()),
            }
        audit = {
            "root": root.as_posix(),
            "images_root": img_root.as_posix(),
            "labels_path": labels_path.as_posix(),
            "split_mode": str(args.split_mode),
            "rows": int(len(df)),
            "unique_files": int(df["file"].nunique()),
            "split_counts": {k: int(v) for k, v in df["split"].astype(str).value_counts().to_dict().items()},
            "glove_type_counts": {k: int(v) for k, v in df["glove_type"].astype(str).value_counts().to_dict().items()},
            "strata": split_by_strata,
            "non_test_evaluable_strata": sorted(
                key for key, item in split_by_strata.items() if int(item.get("test", 0)) <= 0
            ),
            "non_val_strata": sorted(
                key for key, item in split_by_strata.items() if int(item.get("val", 0)) <= 0
            ),
        }
        audit_path = Path(str(args.audit_json))
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote: {audit_path}")


if __name__ == "__main__":
    main()
