from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gdd.core.dataset import load_labels_csv, validate_labels_df
from gdd.core.features import glove_type_features
from gdd.core.glove_type_model import save_glove_type_model, train_glove_type_model
from gdd.core.image_io import read_image, resize_max_side
from gdd.core.preprocess import preprocess
from gdd.core.segmentation import segment_glove


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels.csv", help="Path to labels CSV")
    ap.add_argument("--out", default="gdd/models/glove_type.joblib", help="Output model path")
    args = ap.parse_args()

    df = load_labels_csv(args.labels)
    errs = validate_labels_df(df)
    if errs:
        raise SystemExit("labels.csv validation failed:\n- " + "\n- ".join(errs))

    # Train on non-test images only.
    df = df[df["split"].astype(str).isin(["train", "val"])].copy()
    if df.empty:
        raise SystemExit("No train/val rows in labels.csv")

    x_list: list[np.ndarray] = []
    y_list: list[str] = []
    for _, row in df.iterrows():
        path = str(row["file"])
        gt = str(row["glove_type"])
        img = read_image(path).bgr
        img = resize_max_side(img, max_side=900)
        img_p = preprocess(img)
        seg = segment_glove(img_p)
        feats = glove_type_features(img_p, seg.glove_mask, seg.glove_mask_filled)
        x_list.append(feats)
        y_list.append(gt)

    x = np.stack(x_list, axis=0)
    model = train_glove_type_model(x, y_list)
    out_path = Path(args.out)
    save_glove_type_model(model, out_path)
    print(f"Saved glove type model to {out_path}")


if __name__ == "__main__":
    main()
