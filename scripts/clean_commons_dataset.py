from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gdd.core.image_io import read_image, resize_max_side
from gdd.core.preprocess import preprocess
from gdd.core.segmentation import segment_glove


@dataclass(frozen=True)
class MaskQuality:
    area_frac: float
    extent: float
    border_touch: float
    bbox_aspect: float


def _mask_quality(mask01: np.ndarray) -> MaskQuality:
    h, w = mask01.shape[:2]
    area = float(mask01.sum()) + 1e-6
    area_frac = area / float(h * w)

    x, y, ww, hh = cv2.boundingRect(mask01.astype(np.uint8))
    extent = area / float(ww * hh + 1e-6)
    bbox_aspect = float(max(ww, hh) / float(min(ww, hh) + 1e-6))

    border = max(5, int(round(min(h, w) * 0.04)))
    band = np.zeros_like(mask01, dtype=np.uint8)
    band[:border, :] = 1
    band[-border:, :] = 1
    band[:, :border] = 1
    band[:, -border:] = 1
    border_touch = float((mask01 & band).sum()) / area

    return MaskQuality(area_frac=float(area_frac), extent=float(extent), border_touch=float(border_touch), bbox_aspect=bbox_aspect)


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _save_masked_crop(bgr: np.ndarray, mask01: np.ndarray, out_path: Path, pad_frac: float = 0.10) -> dict:
    h, w = mask01.shape[:2]
    x, y, ww, hh = cv2.boundingRect(mask01.astype(np.uint8))
    pad = int(round(pad_frac * max(ww, hh)))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + ww + pad)
    y1 = min(h, y + hh + pad)

    crop = bgr[y0:y1, x0:x1].copy()
    crop_m = mask01[y0:y1, x0:x1]

    # Set outside-glove pixels to a constant dark background (reduces background bias).
    crop[crop_m == 0] = (0, 0, 0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)
    return {"crop_bbox": [int(x0), int(y0), int(x1), int(y1)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-manifest", default="data/public/commons_raw/manifest.csv", help="Input manifest.csv")
    ap.add_argument("--out-dir", default="data/public/commons_clean", help="Output cleaned dataset dir")
    ap.add_argument("--max-side", type=int, default=900, help="Resize max side before segmentation")
    ap.add_argument("--min-area-frac", type=float, default=0.07)
    ap.add_argument("--max-area-frac", type=float, default=0.80)
    ap.add_argument("--min-extent", type=float, default=0.30)
    ap.add_argument("--max-border-touch", type=float, default=0.18)
    ap.add_argument("--max-bbox-aspect", type=float, default=3.5)
    ap.add_argument("--dedupe", action="store_true", help="Drop exact duplicate files (SHA1)")
    args = ap.parse_args()

    in_manifest = Path(args.in_manifest)
    if not in_manifest.exists():
        raise SystemExit(f"Missing: {in_manifest}")

    df = pd.read_csv(in_manifest)
    out_dir = Path(args.out_dir)
    img_root = out_dir / "images"
    meta_root = out_dir / "meta"
    img_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    rows_out: list[dict] = []
    kept = 0
    skipped = 0

    for _, row in df.iterrows():
        src_path = Path(str(row["file"]))
        glove_type = str(row["glove_type"])
        if not src_path.exists():
            skipped += 1
            continue

        if args.dedupe:
            sha = _sha1_file(src_path)
            if sha in seen_hashes:
                skipped += 1
                continue
            seen_hashes.add(sha)

        try:
            bgr = read_image(src_path).bgr
        except Exception:
            skipped += 1
            continue

        bgr = resize_max_side(bgr, max_side=int(args.max_side))
        bgr = preprocess(bgr)
        seg = segment_glove(bgr)
        mask01 = (seg.glove_mask > 0).astype(np.uint8)
        q = _mask_quality(mask01)

        if not (float(args.min_area_frac) <= q.area_frac <= float(args.max_area_frac)):
            skipped += 1
            continue
        if q.extent < float(args.min_extent):
            skipped += 1
            continue
        if q.border_touch > float(args.max_border_touch):
            skipped += 1
            continue
        if q.bbox_aspect > float(args.max_bbox_aspect):
            skipped += 1
            continue

        # Save masked crop
        out_name = f"{glove_type}_{src_path.stem}.png"
        out_path = img_root / glove_type / out_name
        extra = _save_masked_crop(bgr, mask01, out_path)

        meta = {
            "source_file": src_path.as_posix(),
            "glove_type": glove_type,
            "segmentation_method": seg.method,
            "mask_quality": q.__dict__,
            **extra,
            "license_short": str(row.get("license_short", "")),
            "license_url": str(row.get("license_url", "")),
            "page_url": str(row.get("page_url", "")),
            "title": str(row.get("title", "")),
            "artist": str(row.get("artist", "")),
            "credit": str(row.get("credit", "")),
        }
        (meta_root / glove_type).mkdir(parents=True, exist_ok=True)
        (meta_root / glove_type / (out_name + ".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        rows_out.append(
            {
                "file": out_path.as_posix(),
                "glove_type": glove_type,
                "source": "Wikimedia Commons",
                "title": str(row.get("title", "")),
                "page_url": str(row.get("page_url", "")),
                "license_short": str(row.get("license_short", "")),
                "license_url": str(row.get("license_url", "")),
                "artist": str(row.get("artist", "")),
                "credit": str(row.get("credit", "")),
                "seg_method": seg.method,
                "area_frac": q.area_frac,
                "extent": q.extent,
                "border_touch": q.border_touch,
            }
        )
        kept += 1

    out_manifest = out_dir / "manifest.csv"
    pd.DataFrame(rows_out).to_csv(out_manifest, index=False)
    (out_dir / "SOURCE.md").write_text(
        "\n".join(
            [
                "# Cleaned Commons dataset",
                "",
                "This dataset is derived from Wikimedia Commons images (open-licensed) and filtered automatically using glove-segmentation mask quality checks.",
                "",
                "Files:",
                "- `manifest.csv`: cleaned manifest + attribution + license links",
                "- `images/<type>/...`: masked + cropped images (background outside glove set to black)",
                "- `meta/<type>/*.json`: per-image metadata, including segmentation quality metrics",
                "",
                "Note: cleaning improves consistency for training glove-type classification, but it can also remove hard/realistic cases.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Kept {kept} / {len(df)} (skipped {skipped})")
    print(f"Wrote: {out_manifest}")


if __name__ == "__main__":
    main()
