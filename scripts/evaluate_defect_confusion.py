from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gdd.core.dataset import load_labels_csv, parse_defect_labels, validate_labels_df
from gdd.core.image_io import read_image, resize_max_side
from gdd.core.labels import DEFECT_LABELS
from gdd.core.pipeline import GDDPipeline


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


PRIORITY_GROUPS = [
    ("hole", "tear"),
    ("missing_finger", "extra_fingers"),
    ("improper_roll", "incomplete_beading"),
    ("damaged_by_fold", "wrinkles_dent", "inside_out"),
    ("discoloration", "stain_dirty", "spotting", "plastic_contamination"),
]


def _load_tuned_thresholds(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _threshold_for(
    label: str,
    glove_type: str,
    tuned: dict,
    min_struct_score: float,
    min_surface_score: float,
) -> float:
    glove_type = str(glove_type).strip().lower()
    per_type = tuned.get("per_type", {}) if isinstance(tuned, dict) else {}
    if isinstance(per_type, dict):
        gt_map = per_type.get(glove_type, {})
        if isinstance(gt_map, dict) and label in gt_map:
            return float(gt_map[label])
    if label in STRUCTURAL_LABELS:
        return float(min_struct_score)
    return float(min_surface_score)


def _single_label_from_scores(
    scores: dict[str, float],
    glove_type: str,
    tuned: dict,
    min_struct_score: float,
    min_surface_score: float,
    selection_mode: str = "max_score",
) -> str:
    valid: list[tuple[str, float, float]] = []
    for label in DEFECT_LABELS:
        sc = float(scores.get(label, 0.0))
        thr = _threshold_for(label, glove_type, tuned, min_struct_score, min_surface_score)
        if sc >= thr:
            valid.append((label, sc, thr))
    if not valid:
        return "normal"

    mode = str(selection_mode).strip().lower()
    if mode == "priority":
        valid_map = {lab: (sc, thr) for lab, sc, thr in valid}
        for group in PRIORITY_GROUPS:
            candidates = [lab for lab in group if lab in valid_map]
            if candidates:
                candidates.sort(key=lambda lab: float(valid_map[lab][0]), reverse=True)
                return str(candidates[0])
        return str(max(valid, key=lambda x: x[1])[0])

    if mode == "margin":
        best = max(valid, key=lambda x: (float(x[1] - x[2]), float(x[1])))
        return str(best[0])

    # Default: choose highest confidence among threshold-passing labels.
    best = max(valid, key=lambda x: (float(x[1]), float(x[1] - x[2])))
    return str(best[0])


def _plot_and_save_confusion(df: pd.DataFrame, labels: list[str], out_path: Path) -> None:
    if df.empty:
        return
    cm = confusion_matrix(df["true_label"].tolist(), df["pred_label"].tolist(), labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig = disp.plot(xticks_rotation=30).figure_
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/ai_generated/labels.csv", help="Path to labels CSV")
    ap.add_argument(
        "--splits",
        default="all",
        help="Comma-separated splits to evaluate (e.g. test,val) or 'all' for all rows",
    )
    ap.add_argument("--out-csv", default="results/defect_confusion_predictions.csv", help="Per-image predictions CSV")
    ap.add_argument("--out-dir", default="results", help="Output folder for confusion matrix PNGs")
    ap.add_argument("--max-side", type=int, default=900, help="Resize max side before inference")
    ap.add_argument("--min-struct-score", type=float, default=0.65)
    ap.add_argument("--min-surface-score", type=float, default=0.85)
    ap.add_argument("--thresholds-json", default="gdd/models/defect_thresholds.json", help="Optional tuned thresholds JSON")
    ap.add_argument("--selection-mode", choices=["max_score", "margin", "priority"], default="margin", help="Single-label selection rule when multiple labels pass threshold")
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

    tuned = _load_tuned_thresholds(Path(str(args.thresholds_json)))
    pipeline = GDDPipeline.load_default()

    rows: list[dict] = []
    for _, row in eval_df.iterrows():
        path = str(row["file"])
        glove_type_true = str(row.get("glove_type", "unknown"))
        img = read_image(path).bgr
        img = resize_max_side(img, max_side=int(args.max_side))
        res = pipeline.infer(img, force_glove_type=glove_type_true)

        score_map = {label: 0.0 for label in DEFECT_LABELS}
        for d in res.defects:
            lab = str(d.label)
            if lab not in score_map:
                continue
            score_map[lab] = max(float(score_map.get(lab, 0.0)), float(d.score))

        true_labs = parse_defect_labels(str(row.get("defect_labels", "")))
        true_label = str(true_labs[0]) if len(true_labs) == 1 else ("normal" if len(true_labs) == 0 else str(sorted(true_labs)[0]))
        pred_label = _single_label_from_scores(
            score_map,
            glove_type=glove_type_true,
            tuned=tuned,
            min_struct_score=float(args.min_struct_score),
            min_surface_score=float(args.min_surface_score),
            selection_mode=str(args.selection_mode),
        )
        rows.append(
            {
                "file": path,
                "glove_type": glove_type_true,
                "true_label": true_label,
                "pred_label": pred_label,
                "true_label_count": int(len(true_labs)),
                **{f"score_{label}": float(score_map.get(label, 0.0)) for label in DEFECT_LABELS},
            }
        )

    out_df = pd.DataFrame(rows)
    out_csv = Path(str(args.out_csv))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    out_dir = Path(str(args.out_dir))
    class_labels = ["normal"] + list(DEFECT_LABELS)
    _plot_and_save_confusion(out_df, class_labels, out_dir / "defect_confusion_overall.png")

    for gt in ("latex", "leather", "fabric"):
        sub = out_df[out_df["glove_type"].astype(str) == gt].copy()
        _plot_and_save_confusion(sub, class_labels, out_dir / f"defect_confusion_{gt}.png")

    print(f"Wrote: {out_csv}")
    print(f"Selection mode: {args.selection_mode}")
    print(f"Wrote: {out_dir / 'defect_confusion_overall.png'}")
    print(f"Wrote: {out_dir / 'defect_confusion_latex.png'}")
    print(f"Wrote: {out_dir / 'defect_confusion_leather.png'}")
    print(f"Wrote: {out_dir / 'defect_confusion_fabric.png'}")


if __name__ == "__main__":
    main()
