from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from gdd.core.dataset import load_labels_csv, validate_labels_df
from gdd.core.features import glove_type_features
from gdd.core.glove_type_model import save_glove_type_model, train_glove_type_model
from gdd.core.image_io import read_image, resize_max_side
from gdd.core.preprocess import preprocess
from gdd.core.segmentation import segment_glove


def _manifest_audit(df, ai_only: bool = True) -> list[str]:
    errs: list[str] = []
    if "file" not in df.columns:
        errs.append("labels.csv missing 'file' column")
        return errs

    files = df["file"].dropna().astype(str)
    dup_counts = Counter(files.tolist())
    dups = sorted([f for f, c in dup_counts.items() if int(c) > 1])
    if dups:
        errs.append(f"Duplicate file rows in manifest: {len(dups)}")

    missing = sorted([f for f in files.tolist() if not Path(f).exists()])
    if missing:
        errs.append(f"Missing files referenced by manifest: {len(missing)}")

    if bool(ai_only):
        bad_roots = sorted([f for f in files.tolist() if not f.startswith("data/ai_generated/images/")])
        if bad_roots:
            errs.append(f"Non-ai_generated image rows present: {len(bad_roots)}")
    return errs


def _extract_feature_rows(df, max_side: int) -> tuple[np.ndarray, list[str], list[str]]:
    x_list: list[np.ndarray] = []
    y_list: list[str] = []
    files: list[str] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting glove-type features", unit="img"):
        path = str(row["file"])
        gt = str(row["glove_type"])
        img = read_image(path).bgr
        img = resize_max_side(img, max_side=int(max_side))
        img_p = preprocess(img)
        seg = segment_glove(img_p)
        feats = glove_type_features(img_p, seg.glove_mask, seg.glove_mask_filled)
        x_list.append(feats)
        y_list.append(gt)
        files.append(path)
    return np.stack(x_list, axis=0), y_list, files


def _print_eval(model, x: np.ndarray, y: list[str], files: list[str], split_name: str) -> None:
    conf: dict[str, Counter] = defaultdict(Counter)
    wrong: list[tuple[str, str, str, float]] = []
    correct = 0
    for feats, gt, path in zip(x, y, files):
        pred, score = model.predict(feats)
        conf[str(gt)][str(pred)] += 1
        if str(pred) == str(gt):
            correct += 1
        else:
            wrong.append((Path(path).name, str(gt), str(pred), float(score)))

    total = len(y)
    acc = (float(correct) / float(total)) if total > 0 else 0.0
    print(f"[eval] {split_name}: {correct}/{total} = {acc:.4f}")
    print(f"[eval] {split_name} confusion: " + str({k: dict(v) for k, v in conf.items()}))
    if wrong:
        print(f"[eval] {split_name} mistakes:")
        for name, gt, pred, score in wrong[:15]:
            print(f"  - {name}: {gt} -> {pred} ({score:.4f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/ai_generated/labels.csv", help="Path to labels CSV")
    ap.add_argument("--out", default="gdd/models/glove_type.joblib", help="Output model path")
    ap.add_argument("--max-side", type=int, default=450, help="Resize max side before feature extraction (speed/robustness tradeoff)")
    ap.add_argument("--limit", type=int, default=0, help="If >0, cap number of train/val rows (for quick experiments)")
    ap.add_argument("--model", default="logreg", choices=["logreg", "rf"], help="Classifier type")
    ap.add_argument("--balance", action="store_true", help="Downsample classes to the same size (reduces bias on imbalanced public data)")
    ap.add_argument("--max-per-class", type=int, default=0, help="If >0, cap each glove_type to at most N train/val rows")
    ap.add_argument("--allow-non-ai-generated", action="store_true", help="Allow manifest rows outside data/ai_generated/images")
    args = ap.parse_args()

    df = load_labels_csv(args.labels)
    errs = validate_labels_df(df)
    errs.extend(_manifest_audit(df, ai_only=not bool(args.allow_non_ai_generated)))
    if errs:
        raise SystemExit("labels.csv validation failed:\n- " + "\n- ".join(errs))

    split_counts = Counter(df["split"].astype(str).tolist())
    type_counts = Counter(df["glove_type"].astype(str).tolist())
    print("[manifest] labels=", args.labels)
    print("[manifest] rows=", len(df), "files=", int(df["file"].astype(str).nunique()))
    print("[manifest] split_counts=", dict(split_counts))
    print("[manifest] glove_type_counts=", dict(type_counts))

    # Train on non-test images only.
    df = df[df["split"].astype(str).isin(["train", "val"])].copy()
    if df.empty:
        raise SystemExit("No train/val rows in labels.csv")

    if int(args.max_per_class) > 0:
        cap = int(args.max_per_class)
        df = (
            df.groupby("glove_type", group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), cap), random_state=42))
            .reset_index(drop=True)
        )

    if bool(args.balance):
        # Downsample to the smallest class to reduce imbalance (useful for public-only datasets).
        sizes = df.groupby("glove_type").size().to_dict()
        min_n = int(min(sizes.values())) if sizes else 0
        if min_n >= 2:
            df = (
                df.groupby("glove_type", group_keys=False)
                .apply(lambda g: g.sample(n=min_n, random_state=42))
                .reset_index(drop=True)
            )

    if int(args.limit) > 0 and len(df) > int(args.limit):
        df = df.sample(n=int(args.limit), random_state=42).reset_index(drop=True)

    test_df = load_labels_csv(args.labels)
    test_df = test_df[test_df["split"].astype(str).isin(["test"])].copy()

    x_train, y_train, f_train = _extract_feature_rows(df, max_side=int(args.max_side))
    model = train_glove_type_model(x_train, y_train, model_type=str(args.model))
    _print_eval(model, x_train, y_train, f_train, "trainval")
    if not test_df.empty:
        x_test, y_test, f_test = _extract_feature_rows(test_df, max_side=int(args.max_side))
        _print_eval(model, x_test, y_test, f_test, "test")

    out_path = Path(args.out)
    save_glove_type_model(model, out_path)
    print(f"Saved glove type model to {out_path}")


if __name__ == "__main__":
    main()
