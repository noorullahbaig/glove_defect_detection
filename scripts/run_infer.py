from __future__ import annotations

import argparse
from pathlib import Path

import cv2

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gdd.core.image_io import read_image, resize_max_side
from gdd.core.pipeline import GDDPipeline
from gdd.core.viz import draw_defects, overlay_mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="Path to an input image")
    ap.add_argument("--out", default="results/overlays/infer.png", help="Output overlay image path")
    args = ap.parse_args()

    pipeline = GDDPipeline.load_default()
    img = read_image(args.image).bgr
    img = resize_max_side(img, max_side=1100)
    res = pipeline.infer(img)

    over = overlay_mask(img, res.glove_mask)
    over = draw_defects(over, res.defects)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, over)
    print(f"glove_type={res.glove_type} ({res.glove_type_score:.2f})")
    for d in res.defects:
        loc = f" bbox={d.bbox.as_xyxy()}" if d.bbox else ""
        print(f"- {d.label}: {d.score:.2f}{loc}")
    print(f"Wrote overlay: {args.out}")


if __name__ == "__main__":
    main()
