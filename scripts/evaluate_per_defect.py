from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gdd.core.dataset import load_labels_csv, parse_defect_labels, validate_labels_df
from gdd.core.image_io import read_image, resize_max_side
from gdd.core.labels import DEFECT_LABELS
from gdd.core.pipeline import GDDPipeline


STRUCTURAL_LABELS = {
    "hole",
    "tear",
    "missing_finger",
    "extra_fingers",
    "inside_out",
    "improper_roll",
    "incomplete_beading",
    "damaged_by_fold",
}


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
    ap.add_argument("--out-md", default="results/per_defect_report.md", help="Output Markdown report path")
    ap.add_argument("--out-csv", default="results/per_defect_metrics.csv", help="Output CSV summary path")
    ap.add_argument("--max-side", type=int, default=900, help="Resize max side before inference")
    ap.add_argument("--min-struct-score", type=float, default=0.65)
    ap.add_argument("--min-surface-score", type=float, default=0.85)
    ap.add_argument("--limit-failures", type=int, default=8, help="How many FP/FN examples to list per defect")
    args = ap.parse_args()

    df = load_labels_csv(args.labels)
    errs = validate_labels_df(df)
    if errs:
        raise SystemExit("labels.csv validation failed:\n- " + "\n- ".join(errs))

    test_df = df[df["split"].astype(str) == "test"].copy()
    if test_df.empty:
        raise SystemExit("No test rows in labels.csv (set split=test for some rows).")

    def passes_threshold(label: str, score: float) -> bool:
        if label in STRUCTURAL_LABELS:
            return score >= float(args.min_struct_score)
        return score >= float(args.min_surface_score)

    # Applicability: cuff defects make sense only for latex in this project.
    def applicable_rows(label: str) -> pd.DataFrame:
        if label in {"improper_roll", "incomplete_beading"}:
            return test_df[test_df["glove_type"].astype(str) == "latex"].copy()
        return test_df

    pipeline = GDDPipeline.load_default()

    per_file_rows: list[dict] = []
    summaries: list[DefectEvalSummary] = []
    failure_examples: dict[str, dict[str, list[tuple[str, float]]]] = {}

    for lab in DEFECT_LABELS:
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

            img = read_image(path).bgr
            img = resize_max_side(img, max_side=int(args.max_side))
            res = pipeline.infer(img, focus_only=False, allowed_labels={lab})
            lab_scores = [float(d.score) for d in res.defects if str(d.label) == lab]
            score = float(max(lab_scores)) if lab_scores else 0.0
            pred_pos = bool(passes_threshold(lab, score))

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

    # Markdown report.
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Per-Defect Evaluation Report")
    lines.append("")
    lines.append(f"- Labels file: `{args.labels}`")
    lines.append(f"- Test rows: `{len(test_df)}`")
    lines.append(f"- Scoring thresholds: structural >= `{args.min_struct_score}`, surface >= `{args.min_surface_score}`")
    lines.append(f"- Note: `improper_roll` and `incomplete_beading` are evaluated on `latex` test rows only.")
    lines.append("")
    lines.append("## Summary (sorted by weakest F1)")
    lines.append("")
    lines.append(sm_df.to_markdown(index=False))
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

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {per_file_path}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()

