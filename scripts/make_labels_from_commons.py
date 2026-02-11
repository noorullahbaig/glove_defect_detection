from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/public/commons/manifest.csv", help="Commons manifest.csv path")
    ap.add_argument("--out", default="data/commons_labels.csv", help="Output labels CSV")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    df = pd.read_csv(manifest_path)
    if "file" not in df.columns or "glove_type" not in df.columns:
        raise SystemExit("manifest.csv must contain at least: file, glove_type")

    rng = np.random.default_rng(int(args.seed))
    rows = []

    for gt, gdf in df.groupby("glove_type"):
        files = gdf["file"].astype(str).tolist()
        if not files:
            continue
        idx = np.arange(len(files))
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(float(args.train_frac) * n))
        n_val = int(round(float(args.val_frac) * n))
        n_train = max(1, min(n - 2 if n >= 3 else n - 1, n_train))
        n_val = max(0, min(n - n_train - 1, n_val))

        for i, j in enumerate(idx):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            rows.append(
                {
                    "file": files[int(j)],
                    "glove_type": str(gt),
                    "defect_labels": "",
                    "split": split,
                    "lighting": "unknown",
                    "background": "unknown",
                    "notes": f"source=commons manifest={manifest_path.as_posix()}",
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote: {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

