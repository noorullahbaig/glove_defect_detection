from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gdd.core.dataset import load_labels_csv, parse_defect_labels, validate_labels_df
from gdd.core.image_io import read_image, resize_max_side
from gdd.core.pipeline import GDDPipeline


def _f1(p: float, r: float) -> float:
    return (2 * p * r) / (p + r + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels.csv", help="Path to labels CSV")
    ap.add_argument("--min-struct-score", type=float, default=0.65, help="Min score for structural defects")
    ap.add_argument("--min-surface-score", type=float, default=0.85, help="Min score for surface/color defects")
    args = ap.parse_args()

    df = load_labels_csv(args.labels)
    errs = validate_labels_df(df)
    if errs:
        raise SystemExit("labels.csv validation failed:\n- " + "\n- ".join(errs))

    test_df = df[df["split"].astype(str) == "test"].copy()
    if test_df.empty:
        raise SystemExit("No test rows in labels.csv (set split=test for some rows).")

    # Auto-detect whether defect labels exist in this labels file.
    # If all defect labels are empty, we skip defect metrics (otherwise everything becomes FP).
    has_any_defects = False
    if "defect_labels" in df.columns:
        for v in df["defect_labels"].astype(str).fillna("").tolist():
            if parse_defect_labels(v):
                has_any_defects = True
                break

    pipeline = GDDPipeline.load_default()

    y_true_gt: list[str] = []
    y_pred_gt: list[str] = []

    # Multi-label defect metrics.
    label_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    rows_out = []
    structural_labels = {
        "hole",
        "tear",
        "missing_finger",
        "extra_fingers",
        "inside_out",
        "improper_roll",
        "incomplete_beading",
        "damaged_by_fold",
    }

    def passes_threshold(label: str, score: float) -> bool:
        if label in structural_labels:
            return score >= float(args.min_struct_score)
        return score >= float(args.min_surface_score)

    for _, row in test_df.iterrows():
        path = str(row["file"])
        true_gt = str(row["glove_type"])
        true_def = set(parse_defect_labels(str(row.get("defect_labels", ""))))

        img = read_image(path).bgr
        img = resize_max_side(img, max_side=1000)
        res = pipeline.infer(img)

        pred_gt = res.glove_type
        pred_def = {d.label for d in res.defects if passes_threshold(str(d.label), float(d.score))}
        # If the pipeline has no trained glove-type model loaded, it returns "unknown".
        # In that case, fall back to an explicit "unknown" class in eval instead of
        # skewing the confusion matrix with missing labels.
        if pred_gt == "unknown":
            pred_gt = "unknown"

        y_true_gt.append(true_gt)
        y_pred_gt.append(pred_gt)

        if has_any_defects:
            for lab in sorted(true_def | pred_def):
                if lab in true_def and lab in pred_def:
                    label_stats[lab]["tp"] += 1
                elif lab in pred_def and lab not in true_def:
                    label_stats[lab]["fp"] += 1
                elif lab in true_def and lab not in pred_def:
                    label_stats[lab]["fn"] += 1

        rows_out.append(
            {
                "file": path,
                "true_glove_type": true_gt,
                "pred_glove_type": pred_gt,
                "true_defects": "|".join(sorted(true_def)),
                "pred_defects": "|".join(sorted(pred_def)),
            }
        )

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "eval_results.csv"
    pd.DataFrame(rows_out).to_csv(out_csv, index=False)

    # Confusion matrix for glove type (if model is trained; else 'unknown' dominates).
    labels = sorted(set(y_true_gt) | set(y_pred_gt))
    cm = confusion_matrix(y_true_gt, y_pred_gt, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig = disp.plot(xticks_rotation=30).figure_
    fig.tight_layout()
    fig_path = out_dir / "glove_type_confusion_matrix.png"
    fig.savefig(fig_path, dpi=180)

    if has_any_defects:
        # Defect metrics summary.
        metrics_rows = []
        for lab, s in sorted(label_stats.items()):
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            p = tp / (tp + fp + 1e-9)
            r = tp / (tp + fn + 1e-9)
            metrics_rows.append({"label": lab, "precision": p, "recall": r, "f1": _f1(p, r), "tp": tp, "fp": fp, "fn": fn})
        mdf = pd.DataFrame(metrics_rows).sort_values(["f1", "label"], ascending=[False, True])
        m_path = out_dir / "defect_metrics.csv"
        mdf.to_csv(m_path, index=False)
        print(f"Wrote: {m_path}")
    else:
        print("Skipped defect metrics (no defect_labels present in labels file).")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {fig_path}")


if __name__ == "__main__":
    main()
