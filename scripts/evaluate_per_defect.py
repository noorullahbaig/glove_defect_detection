from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gdd.core.defect_detectors import detect_defects
from gdd.core.dataset import load_labels_csv, parse_defect_labels, validate_labels_df
from gdd.core.image_io import read_image, resize_max_side
from gdd.core.labels import DEFECT_LABELS
from gdd.core.pipeline import GDDPipeline
from gdd.core.preprocess import preprocess
from gdd.core.segmentation import segment_glove
from gdd.core.viz import draw_defects, overlay_mask


STRUCTURAL_LABELS = {
    "hole",
    "tear",
    "wrinkles_dent",
    "missing_finger",
    "extra_fingers",
    "inside_out",
    "improper_roll",
    "incomplete_beading",
    "damaged_by_fold",
}


def _glove_type_from_path(path: str) -> str:
    p = Path(path)
    parts = [str(x).strip().lower() for x in p.parts]
    for gt in ("latex", "leather", "fabric"):
        if gt in parts:
            return gt
    return "unknown"


def _f1(p: float, r: float) -> float:
    return (2 * p * r) / (p + r + 1e-9)


@dataclass(frozen=True)
class DefectEvalSummary:
    label: str
    n_pos: int
    n_neg: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    avg_score_pos: float
    avg_score_neg: float


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/ai_generated/labels.csv", help="Path to labels CSV")
    ap.add_argument(
        "--splits",
        default="all",
        help="Comma-separated splits to evaluate (e.g. test,val) or 'all' for all rows",
    )
    ap.add_argument("--out-md", default="results/per_defect_report.md", help="Output Markdown report path")
    ap.add_argument("--out-csv", default="results/per_defect_metrics.csv", help="Output CSV summary path")
    ap.add_argument("--max-side", type=int, default=900, help="Resize max side before inference")
    ap.add_argument("--min-struct-score", type=float, default=0.65)
    ap.add_argument("--min-surface-score", type=float, default=0.85)
    ap.add_argument("--thresholds-json", default="", help="Optional tuned thresholds JSON from scripts/tune_thresholds.py")
    ap.add_argument("--limit-failures", type=int, default=8, help="How many FP/FN examples to list per defect")
    ap.add_argument("--write-overlays", action="store_true", help="Write overlay images for top FP/FN cases under results/overlays/per_defect/")
    ap.add_argument("--max-overlays", type=int, default=4, help="Max FP and FN overlays per defect (each)")
    args = ap.parse_args()

    df = load_labels_csv(args.labels)
    errs = validate_labels_df(df)
    if errs:
        raise SystemExit("labels.csv validation failed:\n- " + "\n- ".join(errs))

    split_arg = str(args.splits).strip().lower()
    if split_arg == "all":
        eval_df = df.copy()
    else:
        wanted_splits = {s.strip() for s in split_arg.split(",") if s.strip()}
        eval_df = df[df["split"].astype(str).isin(wanted_splits)].copy()
    if eval_df.empty:
        raise SystemExit(f"No rows selected for splits='{args.splits}'.")

    tuned_thresholds: dict = {}
    if str(args.thresholds_json).strip():
        p_thr = Path(str(args.thresholds_json))
        if p_thr.exists():
            try:
                tuned_thresholds = json.loads(p_thr.read_text(encoding="utf-8"))
            except Exception:
                tuned_thresholds = {}

    def passes_threshold(label: str, score: float, glove_type: str) -> bool:
        gt = str(glove_type).strip().lower()
        per_type = tuned_thresholds.get("per_type", {}) if isinstance(tuned_thresholds, dict) else {}
        if isinstance(per_type, dict):
            gt_map = per_type.get(gt, {})
            if isinstance(gt_map, dict) and label in gt_map:
                try:
                    return float(score) >= float(gt_map[label])
                except Exception:
                    pass
        if label in STRUCTURAL_LABELS:
            return float(score) >= float(args.min_struct_score)
        return float(score) >= float(args.min_surface_score)

    # Applicability: cuff defects make sense only for latex in this project.
    def applicable_rows(label: str) -> pd.DataFrame:
        if label in {"improper_roll", "incomplete_beading"}:
            return eval_df[eval_df["glove_type"].astype(str) == "latex"].copy()
        return eval_df

    # Precompute preprocess + segmentation once per image to make per-defect runs fast.
    pipeline = GDDPipeline.load_default()
    cache: dict[str, dict] = {}
    for _, row in eval_df.iterrows():
        path = str(row["file"])
        glove_type = str(row.get("glove_type", "unknown")).strip().lower()
        img = read_image(path).bgr
        img = resize_max_side(img, max_side=int(args.max_side))
        bgr_p = preprocess(img)
        seg_cfg = pipeline.get_profile_seg_cfg(glove_type)
        seg = segment_glove(bgr_p, cfg=seg_cfg)
        cache[path] = {
            "bgr_p": bgr_p,
            "glove_mask": seg.glove_mask,
            "glove_mask_filled": seg.glove_mask_filled,
        }

    per_file_rows: list[dict] = []
    summaries: list[DefectEvalSummary] = []
    failure_examples: dict[str, dict[str, list[tuple[str, float]]]] = {}

    for lab in DEFECT_LABELS:
        print(f"[eval] defect={lab}", flush=True)
        tdf = applicable_rows(lab)
        if tdf.empty:
            continue

        scores_pos: list[float] = []
        scores_neg: list[float] = []
        tp = fp = fn = tn = 0
        fps: list[tuple[str, float]] = []
        fns: list[tuple[str, float]] = []

        for _, row in tdf.iterrows():
            path = str(row["file"])
            true_def = set(parse_defect_labels(str(row.get("defect_labels", ""))))
            is_pos = (lab in true_def)

            it = cache.get(path)
            if it is None:
                continue

            defects, _anom = detect_defects(
                it["bgr_p"],
                it["glove_mask"],
                it["glove_mask_filled"],
                glove_type=str(row.get("glove_type", "unknown")),
                focus_only=False,
                allowed_labels={lab},
            )
            lab_scores = [float(d.score) for d in defects if str(d.label) == lab]
            score = float(max(lab_scores)) if lab_scores else 0.0
            pred_pos = bool(passes_threshold(lab, score, str(row.get("glove_type", "unknown"))))

            if is_pos:
                scores_pos.append(score)
            else:
                scores_neg.append(score)

            if is_pos and pred_pos:
                tp += 1
            elif (not is_pos) and pred_pos:
                fp += 1
                fps.append((path, score))
            elif is_pos and (not pred_pos):
                fn += 1
                fns.append((path, score))
            else:
                tn += 1

            per_file_rows.append(
                {
                    "defect": lab,
                    "file": path,
                    "glove_type": str(row.get("glove_type", "")),
                    "true": bool(is_pos),
                    "score": score,
                    "pred": bool(pred_pos),
                }
            )

        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = _f1(p, r)
        n_pos = int(tp + fn)
        n_neg = int(fp + tn)
        avg_pos = float(sum(scores_pos) / max(1, len(scores_pos))) if scores_pos else 0.0
        avg_neg = float(sum(scores_neg) / max(1, len(scores_neg))) if scores_neg else 0.0

        summaries.append(
            DefectEvalSummary(
                label=lab,
                n_pos=n_pos,
                n_neg=n_neg,
                tp=int(tp),
                fp=int(fp),
                fn=int(fn),
                tn=int(tn),
                precision=float(p),
                recall=float(r),
                f1=float(f1),
                avg_score_pos=float(avg_pos),
                avg_score_neg=float(avg_neg),
            )
        )

        fps.sort(key=lambda x: x[1], reverse=True)
        fns.sort(key=lambda x: x[1])
        failure_examples[lab] = {
            "fp": fps[: int(args.limit_failures)],
            "fn": fns[: int(args.limit_failures)],
        }

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write CSV summary.
    sm_df = pd.DataFrame([s.__dict__ for s in summaries]).sort_values(["f1", "label"], ascending=[True, True])
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sm_df.to_csv(out_csv, index=False)

    # Write detailed per-file results too (useful for debugging).
    per_file = pd.DataFrame(per_file_rows)
    per_file_path = out_dir / "per_defect_per_file.csv"
    per_file.to_csv(per_file_path, index=False)

    if bool(args.write_overlays):
        ov_dir = out_dir / "overlays" / "per_defect"
        ov_dir.mkdir(parents=True, exist_ok=True)

        def _badge(bgr, text: str) -> None:
            import cv2
            cv2.rectangle(bgr, (10, 10), (10 + 760, 10 + 44), (0, 0, 0), -1)
            cv2.putText(bgr, text, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        for lab in DEFECT_LABELS:
            ex = failure_examples.get(lab) or {}
            for kind in ("fp", "fn"):
                items = (ex.get(kind) or [])[: int(args.max_overlays)]
                for idx, (pth, sc) in enumerate(items):
                    try:
                        img = read_image(pth).bgr
                    except Exception:
                        continue
                    img = resize_max_side(img, max_side=int(args.max_side))
                    bgr_p = preprocess(img)
                    gt_overlay = _glove_type_from_path(str(pth))
                    seg = segment_glove(bgr_p, cfg=pipeline.get_profile_seg_cfg(gt_overlay))
                    defects, _anom = detect_defects(
                        bgr_p,
                        seg.glove_mask,
                        seg.glove_mask_filled,
                        glove_type=gt_overlay,
                        focus_only=False,
                        allowed_labels={lab},
                    )
                    over = overlay_mask(img, seg.glove_mask)
                    boxed = [d for d in defects if d.bbox is not None]
                    over = draw_defects(over, boxed)
                    _badge(over, f"{lab} {kind.upper()} score={sc:.3f} file={Path(pth).name}")
                    outp = ov_dir / lab / kind
                    outp.mkdir(parents=True, exist_ok=True)
                    import cv2
                    cv2.imwrite(str(outp / f"{idx:02d}.png"), over)

    # Markdown report.
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
        view = df[cols].copy()
        # Basic, dependency-free markdown table.
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = []
        for _, r in view.iterrows():
            vals = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    vals.append(f"{v:.3f}")
                else:
                    vals.append(str(v))
            rows.append("| " + " | ".join(vals) + " |")
        return "\n".join([header, sep] + rows)

    lines: list[str] = []
    lines.append("# Per-Defect Evaluation Report")
    lines.append("")
    lines.append(f"- Labels file: `{args.labels}`")
    lines.append(f"- Evaluated rows: `{len(eval_df)}`")
    lines.append(f"- Splits used: `{args.splits}`")
    lines.append(f"- Scoring thresholds: structural >= `{args.min_struct_score}`, surface >= `{args.min_surface_score}`")
    if tuned_thresholds:
        lines.append(f"- Tuned thresholds JSON: `{args.thresholds_json}`")
    lines.append(f"- Note: `improper_roll` and `incomplete_beading` are evaluated on `latex` rows only.")
    lines.append("")
    lines.append("## Summary (sorted by weakest F1)")
    lines.append("")
    lines.append(
        _md_table(
            sm_df,
            cols=[
                "label",
                "n_pos",
                "n_neg",
                "tp",
                "fp",
                "fn",
                "tn",
                "precision",
                "recall",
                "f1",
                "avg_score_pos",
                "avg_score_neg",
            ],
        )
    )
    lines.append("")
    lines.append("## Failure Examples (Top FP/FN per defect)")
    lines.append("")
    for s in sorted(summaries, key=lambda x: (x.f1, x.label)):
        lab = s.label
        lines.append(f"### `{lab}`")
        lines.append(f"- Positives in eval: `{s.n_pos}` Negatives in eval: `{s.n_neg}`")
        ex = failure_examples.get(lab) or {}
        fps = ex.get("fp") or []
        fns = ex.get("fn") or []
        if fps:
            lines.append("- False positives (highest scores):")
            for pth, sc in fps:
                lines.append(f"  - `{pth}` score={sc:.3f}")
        if fns:
            lines.append("- False negatives (lowest scores):")
            for pth, sc in fns:
                lines.append(f"  - `{pth}` score={sc:.3f}")
        if not fps and not fns:
            lines.append("- No FP/FN examples recorded.")
        lines.append("")

    lines.append("## Recommendations (Data To Add / Prompt Fixes)")
    lines.append("")
    lines.append("This dataset is small, so many defects have only 1–3 positive samples. For stable metrics, target **at least 6 images per (glove_type, defect)**,")
    lines.append("so that each defect has enough positive coverage for robust threshold tuning and evaluation.")
    lines.append("")
    for s in sorted(summaries, key=lambda x: (x.f1, x.label)):
        lab = s.label
        recs: list[str] = []
        if s.n_pos < 3:
            recs.append(f"Add more positives: currently n_pos={s.n_pos} in test. Generate 6–10 total images for `{lab}` across glove types.")
        if lab == "tear":
            recs.append("Prompt tweak: make the tear show background through the rip (a true void). The current tear detector relies on missing mask pixels inside the glove silhouette.")
        if lab in {"extra_fingers", "missing_finger"}:
            recs.append("Prompt tweak: top-down, glove lying flat, high-contrast background, and make the finger count visually unambiguous in the silhouette (no motion blur, no occlusion).")
        if lab == "inside_out":
            recs.append("Prompt tweak: emphasize turned-inside-out seams/inner structure clearly visible on the outside; avoid smooth exterior-looking renders.")
        if lab in {"stain_dirty", "spotting", "plastic_contamination"}:
            recs.append("Add more `normal` images too (per glove type). These defects are prone to false positives on AI textures; more clean negatives help tuning and evaluation.")
        if lab in {"improper_roll", "incomplete_beading"}:
            recs.append("Latex-only: generate only on latex gloves, with the cuff region large in frame (glove opening visible).")

        if recs:
            lines.append(f"### `{lab}`")
            for r in recs:
                lines.append(f"- {r}")
            lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {per_file_path}")
    print(f"Wrote: {out_md}")
    if bool(args.write_overlays):
        print(f"Wrote overlays under: {out_dir / 'overlays' / 'per_defect'}")


if __name__ == "__main__":
    main()
