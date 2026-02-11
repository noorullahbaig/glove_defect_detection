from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True, help="One or more manifest.csv paths to merge")
    ap.add_argument("--out", default="data/public/public_raw_manifest.csv", help="Output merged manifest.csv path")
    args = ap.parse_args()

    dfs = []
    for p in args.manifests:
        mp = Path(p)
        if not mp.exists():
            raise SystemExit(f"Missing manifest: {mp}")
        df = pd.read_csv(mp)
        if "file" not in df.columns or "glove_type" not in df.columns:
            raise SystemExit(f"Invalid manifest (needs file, glove_type): {mp}")
        dfs.append(df)

    out_df = pd.concat(dfs, ignore_index=True)
    out_df["file"] = out_df["file"].astype(str)
    out_df = out_df.drop_duplicates(subset=["file"]).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()

