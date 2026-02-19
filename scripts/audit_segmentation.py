from __future__ import annotations

import argparse
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
from gdd.core.segmentation import segment_glove_debug
from gdd.core.viz import overlay_mask


@dataclass(frozen=True)
class ProxyMetrics:
    area_frac: float
    extent: float
    border_touch: float
    edge_align: float


def _border_touch(mask01: np.ndarray) -> float:
    m = (mask01 > 0).astype(np.uint8)
    area = float(m.sum()) + 1e-6
    h, w = m.shape[:2]
    band = max(5, int(round(min(h, w) * 0.04)))
    border = np.zeros_like(m, dtype=np.uint8)
    border[:band, :] = 1
    border[-band:, :] = 1
    border[:, :band] = 1
    border[:, -band:] = 1
    return float((m & border).sum()) / float(area)


def _edge_align(mask01: np.ndarray, edges01: np.ndarray) -> float:
    m = (mask01 > 0).astype(np.uint8)
    if m.sum() < 50:
        return 0.0
    k = 5
    dil = cv2.dilate(m, np.ones((k, k), np.uint8), iterations=1)
    ero = cv2.erode(m, np.ones((k, k), np.uint8), iterations=1)
    band = (dil > 0).astype(np.uint8) - (ero > 0).astype(np.uint8)
    denom = float(band.sum()) + 1e-6
    return float(((band > 0) & (edges01 > 0)).sum()) / denom


def _proxy_metrics(mask_u8: np.ndarray, edges_u8: np.ndarray) -> ProxyMetrics:
    mask01 = (mask_u8 > 0).astype(np.uint8)
    h, w = mask01.shape[:2]
    area = float(mask01.sum())
    area_frac = area / float(h * w + 1e-6)
    x, y, ww, hh = cv2.boundingRect(mask01.astype(np.uint8))
    extent = float(area) / float(ww * hh + 1e-6) if ww > 0 and hh > 0 else 0.0
    bt = _border_touch(mask01)
    ea = _edge_align(mask01, (edges_u8 > 0).astype(np.uint8))
    return ProxyMetrics(area_frac=float(area_frac), extent=float(extent), border_touch=float(bt), edge_align=float(ea))


def _badness(m: ProxyMetrics) -> float:
    bad = 0.0
    bad += 4.0 * max(0.0, 0.06 - float(m.edge_align))
    bad += 4.0 * max(0.0, float(m.border_touch) - 0.12)
    bad += 3.0 * max(0.0, 0.05 - float(m.area_frac))
    bad += 3.0 * max(0.0, float(m.area_frac) - 0.80)
    bad += 2.0 * max(0.0, 0.18 - float(m.extent))
    return float(bad)


def _iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    out: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        out.extend(path.rglob(ext))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Image file or folder to audit")
    ap.add_argument("--out-dir", default="results/seg_audit", help="Output folder for overlays/CSVs")
    ap.add_argument("--max-images", type=int, default=250)
    ap.add_argument("--max-side", type=int, default=1100)
    ap.add_argument("--write-overlays", action="store_true", help="Write overlay PNGs for the worst cases")
    ap.add_argument("--worst-n", type=int, default=40, help="How many worst cases to export when --write-overlays is set")
    args = ap.parse_args()

    in_path = Path(str(args.path))
    if not in_path.exists():
        raise SystemExit(f"Not found: {in_path}")

    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = _iter_images(in_path)[: int(args.max_images)]
    rows: list[dict] = []

    for p in paths:
        try:
            bgr = read_image(p).bgr
        except Exception:
            continue

        bgr = resize_max_side(bgr, max_side=int(args.max_side))
        bgr_p = preprocess(bgr)

        try:
            res, dbg = segment_glove_debug(bgr_p)
        except Exception:
            continue

        pm = _proxy_metrics(res.glove_mask, dbg.edges_u8)
        bad = _badness(pm)

        top = dbg.scored[0] if dbg.scored else {}
        rows.append(
            {
                "file": str(p),
                "seg_method": res.method,
                "bg_is_white": bool(dbg.bg_is_white),
                "bg_is_uniform": bool(dbg.bg_is_uniform),
                "area_frac": pm.area_frac,
                "extent": pm.extent,
                "border_touch": pm.border_touch,
                "edge_align": pm.edge_align,
                "badness": bad,
                "top_candidate": str(top.get("method", "")),
                "top_score": float(top.get("score", 0.0)) if isinstance(top.get("score", 0.0), (int, float)) else 0.0,
                "top_bg_sep_u8": float(top.get("bg_sep_u8", 0.0)) if isinstance(top.get("bg_sep_u8", 0.0), (int, float)) else 0.0,
                "top_edge_align": float(top.get("edge_align", 0.0)) if isinstance(top.get("edge_align", 0.0), (int, float)) else 0.0,
                "scored_json": json.dumps(dbg.scored[:5]),
            }
        )

    df = pd.DataFrame(rows).sort_values("badness", ascending=False)
    csv_path = out_dir / "segmentation_audit.csv"
    df.to_csv(csv_path, index=False)

    print(f"Wrote: {csv_path}")
    if df.empty:
        print("No images processed.")
        return

    worst = df.head(int(args.worst_n))
    print("Worst cases:")
    for _, r in worst.iterrows():
        print(f"- bad={r['badness']:.3f} area={r['area_frac']:.3f} extent={r['extent']:.3f} border={r['border_touch']:.3f} edge={r['edge_align']:.3f} :: {r['file']} ({r['seg_method']})")

    if not bool(args.write_overlays):
        return

    export_dir = out_dir / "worst_overlays"
    export_dir.mkdir(parents=True, exist_ok=True)

    for _, r in worst.iterrows():
        p = Path(str(r["file"]))
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        bgr = resize_max_side(bgr, max_side=int(args.max_side))
        bgr_p = preprocess(bgr)
        res, dbg = segment_glove_debug(bgr_p)

        over = overlay_mask(bgr, res.glove_mask)
        heat = cv2.applyColorMap(dbg.d_bg_u8, cv2.COLORMAP_TURBO)
        edges = cv2.cvtColor(dbg.edges_u8, cv2.COLOR_GRAY2BGR)

        h, w = bgr.shape[:2]
        tiles = [
            cv2.resize(bgr, (w, h)),
            cv2.resize(over, (w, h)),
            cv2.resize(heat, (w, h)),
            cv2.resize(edges, (w, h)),
        ]
        top = np.concatenate(tiles[:2], axis=1)
        bot = np.concatenate(tiles[2:], axis=1)
        grid = np.concatenate([top, bot], axis=0)

        out_name = (p.stem[:90] + "_audit.png").replace("/", "_")
        cv2.imwrite(str(export_dir / out_name), grid)

    print(f"Wrote overlays: {export_dir}")


if __name__ == "__main__":
    main()
