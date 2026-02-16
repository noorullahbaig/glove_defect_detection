from __future__ import annotations

import argparse
import random
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
from gdd.core.labels import GLOVE_TYPES


@dataclass(frozen=True)
class SynthConfig:
    max_side: int = 1200
    seed: int = 42


def _sample_background_color(bgr: np.ndarray, glove_mask01: np.ndarray) -> tuple[int, int, int]:
    # Sample from outside the glove; fall back to a dark gray.
    outside = bgr[glove_mask01 == 0]
    if outside.size == 0:
        return (30, 30, 30)
    idx = np.random.choice(outside.shape[0], size=min(5000, outside.shape[0]), replace=False)
    med = np.median(outside[idx], axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def _random_point_in_mask(mask01: np.ndarray) -> tuple[int, int] | None:
    ys, xs = np.where(mask01 > 0)
    if ys.size == 0:
        return None
    i = int(np.random.randint(0, ys.size))
    return int(xs[i]), int(ys[i])


def _draw_hole(bgr: np.ndarray, glove_mask01: np.ndarray) -> tuple[np.ndarray, dict] | None:
    pt = _random_point_in_mask(glove_mask01)
    if pt is None:
        return None
    x, y = pt
    h, w = glove_mask01.shape[:2]
    r = int(round(0.03 * min(h, w)))
    r = max(8, min(r, 45))
    bg = _sample_background_color(bgr, glove_mask01)

    out = bgr.copy()
    cv2.circle(out, (x, y), r, bg, -1, lineType=cv2.LINE_AA)
    meta = {"x": x, "y": y, "r": r}
    return out, meta


def _draw_tear(bgr: np.ndarray, glove_mask01: np.ndarray) -> tuple[np.ndarray, dict] | None:
    pt = _random_point_in_mask(glove_mask01)
    if pt is None:
        return None
    x, y = pt
    h, w = glove_mask01.shape[:2]
    length = int(round(0.18 * min(h, w)))
    length = max(35, min(length, 160))
    thickness = int(round(length * 0.12))
    thickness = max(6, min(thickness, 25))
    angle = float(np.random.uniform(-80, 80))
    dx = int(round(np.cos(np.deg2rad(angle)) * length / 2))
    dy = int(round(np.sin(np.deg2rad(angle)) * length / 2))
    x1, y1 = x - dx, y - dy
    x2, y2 = x + dx, y + dy
    bg = _sample_background_color(bgr, glove_mask01)

    out = bgr.copy()
    cv2.line(out, (x1, y1), (x2, y2), bg, thickness=thickness, lineType=cv2.LINE_AA)
    meta = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "thickness": thickness}
    return out, meta


def _draw_stain(bgr: np.ndarray, glove_mask01: np.ndarray) -> tuple[np.ndarray, dict] | None:
    pt = _random_point_in_mask(glove_mask01)
    if pt is None:
        return None
    x, y = pt
    h, w = glove_mask01.shape[:2]
    rad = int(round(0.06 * min(h, w)))
    rad = max(18, min(rad, 90))

    out = bgr.copy()
    # Dark brown-ish stain
    color = (int(np.random.randint(10, 60)), int(np.random.randint(30, 90)), int(np.random.randint(80, 140)))
    alpha = float(np.random.uniform(0.25, 0.45))

    overlay = out.copy()
    cv2.circle(overlay, (x, y), rad, color, -1, lineType=cv2.LINE_AA)
    out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0.0)
    meta = {"x": x, "y": y, "r": rad, "alpha": alpha}
    return out, meta


def _draw_spots(bgr: np.ndarray, glove_mask01: np.ndarray) -> tuple[np.ndarray, dict] | None:
    ys, xs = np.where(glove_mask01 > 0)
    if ys.size == 0:
        return None
    out = bgr.copy()
    n = int(np.random.randint(8, 25))
    spots = []
    for _ in range(n):
        i = int(np.random.randint(0, ys.size))
        x, y = int(xs[i]), int(ys[i])
        r = int(np.random.randint(2, 7))
        cv2.circle(out, (x, y), r, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        spots.append({"x": x, "y": y, "r": r})
    return out, {"count": n, "spots": spots[:8]}


def _draw_discoloration(bgr: np.ndarray, glove_mask01: np.ndarray) -> tuple[np.ndarray, dict] | None:
    pt = _random_point_in_mask(glove_mask01)
    if pt is None:
        return None
    x, y = pt
    h, w = glove_mask01.shape[:2]
    rad = int(round(0.10 * min(h, w)))
    rad = max(28, min(rad, 140))

    out = bgr.copy()
    overlay = out.copy()
    # Slight yellow tint
    color = (int(np.random.randint(120, 170)), int(np.random.randint(170, 220)), int(np.random.randint(210, 245)))
    alpha = float(np.random.uniform(0.18, 0.30))
    cv2.circle(overlay, (x, y), rad, color, -1, lineType=cv2.LINE_AA)
    out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0.0)
    return out, {"x": x, "y": y, "r": rad, "alpha": alpha}


DEFECT_DRAWERS = {
    "hole": _draw_hole,
    "tear": _draw_tear,
    "stain_dirty": _draw_stain,
    "spotting": _draw_spots,
    "discoloration": _draw_discoloration,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-dir",
        default="data/public/public_clean",
        help="Input dataset folder (supports both raw collectors and cleaned datasets)",
    )
    ap.add_argument("--out-dir", default="data/public/synth", help="Output folder for synthetic defects")
    ap.add_argument("--per-image", type=int, default=2, help="How many synthetic variants per source image")
    ap.add_argument("--max-per-type", type=int, default=80, help="Max source images per glove type")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    cfg = SynthConfig(seed=int(args.seed))
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for glove_type in GLOVE_TYPES:
        # Support both:
        # - raw collectors: <in_dir>/<type>/images/*.jpg
        # - cleaned datasets: <in_dir>/images/<type>/*.png
        src_dir = in_dir / glove_type / "images"
        if not src_dir.exists():
            src_dir = in_dir / "images" / glove_type
        if not src_dir.exists():
            print(f"Skipping missing: {src_dir}")
            continue

        src_paths = sorted([p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])[: int(args.max_per_type)]

        for src in src_paths:
            img = read_image(src).bgr
            img = resize_max_side(img, max_side=cfg.max_side)
            img_p = preprocess(img)
            seg = segment_glove(img_p)
            mask01 = (seg.glove_mask > 0).astype(np.uint8)
            area = int(mask01.sum())
            h, w = mask01.shape[:2]
            if area < 0.05 * h * w or area > 0.85 * h * w:
                continue

            for j in range(int(args.per_image)):
                defect = random.choice(list(DEFECT_DRAWERS.keys()))
                drawer = DEFECT_DRAWERS[defect]
                out = drawer(img_p, mask01)
                if out is None:
                    continue
                out_img, meta = out
                out_path = out_dir / glove_type / defect / f"{src.stem}_synth_{j:02d}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), out_img)

                rows.append(
                    {
                        "file": str(out_path.as_posix()),
                        "glove_type": glove_type,
                        "defect_labels": defect,
                        "split": args.split,
                        "lighting": "synthetic",
                        "background": "synthetic",
                        "notes": f"synth_from={src.as_posix()} meta={meta}",
                    }
                )

    labels_path = out_dir / "labels.csv"
    pd.DataFrame(rows).to_csv(labels_path, index=False)
    print(f"Wrote: {labels_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
