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
    args = ap.parse_args()

    df = load_labels_csv(args.labels)
    errs = validate_labels_df(df)
    if errs:
        raise SystemExit("labels.csv validation failed:\n- " + "\n- ".join(errs))

    test_df = df[df["split"].astype(str) == "test"].copy()
    if test_df.empty:
        raise SystemExit("No test rows in labels.csv (set split=test for some rows).")

    pipeline = GDDPipeline.load_default()

    y_true_gt: list[str] = []
    y_pred_gt: list[str] = []

    # Multi-label defect metrics.
    label_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    rows_out = []
    for _, row in test_df.iterrows():
        path = str(row["file"])
        true_gt = str(row["glove_type"])
        true_def = set(parse_defect_labels(str(row.get("defect_labels", ""))))

        img = read_image(path).bgr
        img = resize_max_side(img, max_side=1000)
        res = pipeline.infer(img)

        pred_gt = res.glove_type
        pred_def = {d.label for d in res.defects if d.score >= 0.55}
        # If the pipeline has no trained glove-type model loaded, it returns "unknown".
        # In that case, fall back to an explicit "unknown" class in eval instead of
        # skewing the confusion matrix with missing labels.
        if pred_gt == "unknown":
            pred_gt = "unknown"

        y_true_gt.append(true_gt)
        y_pred_gt.append(pred_gt)

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

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {fig_path}")
    print(f"Wrote: {m_path}")


if __name__ == "__main__":
    main()
