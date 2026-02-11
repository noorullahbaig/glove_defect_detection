from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _is_nc_or_nd(lic: str) -> bool:
    s = (lic or "").lower()
    return ("nc" in s) or ("nd" in s)


def _is_permissive(lic: str) -> bool:
    s = (lic or "").strip()
    sl = s.lower()
    # Openverse short codes
    if sl in {"by", "by-sa", "cc0", "pdm"}:
        return True
    # Commons / CC short names
    if s.startswith("CC BY") or s.startswith("CC BY-SA") or s.startswith("CC0"):
        return True
    if s.startswith("Public domain") or s.startswith("PD"):
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-manifest", required=True, help="Input manifest.csv")
    ap.add_argument("--out", required=True, help="Output manifest.csv")
    ap.add_argument("--allow-nc", action="store_true", help="Keep NonCommercial/NoDerivatives items too (default: drop)")
    args = ap.parse_args()

    in_path = Path(args.in_manifest)
    if not in_path.exists():
        raise SystemExit(f"Missing: {in_path}")

    df = pd.read_csv(in_path)
    if "license_short" not in df.columns:
        raise SystemExit("manifest.csv must have a license_short column")

    df["license_short"] = df["license_short"].astype(str)
    df = df[df["license_short"].map(_is_permissive)].copy()
    if not bool(args.allow_nc):
        df = df[~df["license_short"].map(_is_nc_or_nd)].copy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()

