from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gdd.core.dataset import load_labels_csv, validate_labels_df
from gdd.core.features import glove_type_features
from gdd.core.glove_type_model import save_glove_type_model, train_glove_type_model
from gdd.core.image_io import read_image, resize_max_side
from gdd.core.labels import DEFECT_LABELS, GLOVE_TYPES
from gdd.core.preprocess import preprocess
from gdd.core.segmentation import segment_glove


def _extract_feature_rows(df: pd.DataFrame, max_side: int) -> tuple[list[str], list[str], list]:
    feats_out: list = []
    y_out: list[str] = []
    files_out: list[str] = []
    for _, row in df.iterrows():
        path = str(row["file"])
        img = read_image(path).bgr
        img = resize_max_side(img, max_side=int(max_side))
        img_p = preprocess(img)
        seg = segment_glove(img_p)
        feats = glove_type_features(img_p, seg.glove_mask, seg.glove_mask_filled)
        feats_out.append(feats)
        y_out.append(str(row["glove_type"]))
        files_out.append(path)
    return y_out, files_out, feats_out


def _evaluate_classifier(model, feats_list: list, y_true: list[str], files: list[str]) -> tuple[pd.DataFrame, dict]:
    y_pred: list[str] = []
    y_score: list[float] = []
    for feats in feats_list:
        pred, score = model.predict(feats)
        y_pred.append(str(pred))
        y_score.append(float(score))

    labels = list(GLOVE_TYPES)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = [
        {
            "file": path,
            "true_glove_type": true_lab,
            "pred_glove_type": pred_lab,
            "pred_score": float(score),
        }
        for path, true_lab, pred_lab, score in zip(files, y_true, y_pred, y_score, strict=True)
    ]
    metrics = {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f_macro),
        "labels": labels,
        "support_by_class": {
            label: int(sum(1 for value in y_true if str(value) == label)) for label in labels
        },
        "per_class": {
            label: {
                "precision": float(report.get(label, {}).get("precision", 0.0)),
                "recall": float(report.get(label, {}).get("recall", 0.0)),
                "f1": float(report.get(label, {}).get("f1-score", 0.0)),
                "support": int(report.get(label, {}).get("support", 0)),
            }
            for label in labels
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": [[int(x) for x in row] for row in cm.tolist()],
        },
    }
    return pd.DataFrame(rows), metrics


def _run_command(cmd: list[str], workdir: Path) -> None:
    result = subprocess.run(cmd, cwd=str(workdir), check=False)
    if result.returncode != 0:
        raise SystemExit(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def _load_metrics_csv(path: Path, mode_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["label"])
    out = df.copy()
    rename_map = {
        "precision": f"{mode_name}_precision",
        "recall": f"{mode_name}_recall",
        "f1": f"{mode_name}_f1",
        "tp": f"{mode_name}_tp",
        "fp": f"{mode_name}_fp",
        "fn": f"{mode_name}_fn",
        "tn": f"{mode_name}_tn",
        "n_pos": f"{mode_name}_n_pos",
        "n_neg": f"{mode_name}_n_neg",
    }
    keep = ["label"] + [k for k in rename_map if k in out.columns]
    out = out[keep].rename(columns=rename_map)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ai_generated", help="AI-generated dataset root")
    ap.add_argument("--out-dir", default="results/ai_generated_eval", help="Output directory")
    ap.add_argument(
        "--classifier-models",
        default="logreg,rf",
        help="Comma-separated glove-type model candidates",
    )
    ap.add_argument("--max-side", type=int, default=450, help="Max side for glove-type feature extraction")
    ap.add_argument("--python", default=sys.executable, help="Python executable to use for child scripts")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    root = Path(str(args.root))
    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_audit_path = out_dir / "manifest_audit.json"
    labels_path = root / "labels.csv"

    _run_command(
        [
            str(args.python),
            "scripts/init_ai_generated_dataset.py",
            "--root",
            str(root),
            "--split-mode",
            "balanced",
            "--audit-json",
            str(manifest_audit_path),
        ],
        workdir=repo_root,
    )

    df = load_labels_csv(labels_path)
    errs = validate_labels_df(df)
    if errs:
        raise SystemExit("labels.csv validation failed:\n- " + "\n- ".join(errs))

    img_root_prefix = (root / "images").as_posix().rstrip("/") + "/"
    if not df["file"].astype(str).str.startswith(img_root_prefix).all():
        raise SystemExit("Manifest contains non-ai_generated source rows after regeneration.")

    train_df = df[df["split"].astype(str) == "train"].copy()
    val_df = df[df["split"].astype(str) == "val"].copy()
    trainval_df = df[df["split"].astype(str).isin(["train", "val"])].copy()
    test_df = df[df["split"].astype(str) == "test"].copy()
    if train_df.empty or test_df.empty:
        raise SystemExit("AI-generated manifest must contain at least train and test rows.")

    train_y, train_files, train_feats = _extract_feature_rows(train_df, max_side=int(args.max_side))
    val_y, val_files, val_feats = _extract_feature_rows(val_df, max_side=int(args.max_side)) if not val_df.empty else ([], [], [])
    trainval_y, trainval_files, trainval_feats = _extract_feature_rows(trainval_df, max_side=int(args.max_side))
    test_y, test_files, test_feats = _extract_feature_rows(test_df, max_side=int(args.max_side))

    candidate_models = [item.strip() for item in str(args.classifier_models).split(",") if item.strip()]
    if not candidate_models:
        raise SystemExit("No classifier models specified.")

    best_name = candidate_models[0]
    best_model = None
    best_key = (-1.0, -1.0)
    candidate_summaries: dict[str, dict] = {}
    for model_name in candidate_models:
        model = train_glove_type_model(train_feats, train_y, model_type=model_name)
        eval_feats = val_feats if val_feats else train_feats
        eval_y = val_y if val_y else train_y
        eval_files = val_files if val_files else train_files
        _, val_metrics = _evaluate_classifier(model, eval_feats, eval_y, eval_files)
        candidate_summaries[model_name] = val_metrics
        key = (float(val_metrics["macro_f1"]), float(val_metrics["accuracy"]))
        if key > best_key:
            best_key = key
            best_name = model_name
            best_model = model

    final_model = train_glove_type_model(trainval_feats, trainval_y, model_type=best_name)
    model_out = repo_root / "gdd" / "models" / "glove_type.joblib"
    save_glove_type_model(final_model, model_out)

    pred_df, test_metrics = _evaluate_classifier(final_model, test_feats, test_y, test_files)
    pred_path = out_dir / "classifier_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    classifier_metrics = {
        "selected_model": best_name,
        "candidate_validation_metrics": candidate_summaries,
        "final_test_metrics": test_metrics,
    }
    metrics_path = out_dir / "classifier_metrics.json"
    metrics_path.write_text(json.dumps(classifier_metrics, indent=2) + "\n", encoding="utf-8")

    cm = confusion_matrix(test_y, pred_df["pred_glove_type"].astype(str).tolist(), labels=list(GLOVE_TYPES))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(GLOVE_TYPES))
    fig = disp.plot(xticks_rotation=30).figure_
    fig.tight_layout()
    cm_path = out_dir / "classifier_confusion_matrix.png"
    fig.savefig(cm_path, dpi=180)

    modes = [
        {
            "name": "default_ui",
            "per_defect_cmd": [
                str(args.python),
                "scripts/evaluate_per_defect.py",
                "--labels",
                str(labels_path),
                "--splits",
                "test",
                "--out-dir",
                str(out_dir / "default_ui"),
                "--out-md",
                str(out_dir / "default_ui" / "per_defect_report.md"),
                "--out-csv",
                str(out_dir / "default_ui" / "defect_metrics.csv"),
                "--out-csv-by-type",
                str(out_dir / "default_ui" / "defect_metrics_by_type.csv"),
                "--min-struct-score",
                "0.65",
                "--min-surface-score",
                "0.85",
            ],
            "confusion_cmd": [
                str(args.python),
                "scripts/evaluate_defect_confusion.py",
                "--labels",
                str(labels_path),
                "--splits",
                "test",
                "--out-csv",
                str(out_dir / "default_ui" / "defect_confusion_predictions.csv"),
                "--out-dir",
                str(out_dir / "default_ui"),
                "--min-struct-score",
                "0.65",
                "--min-surface-score",
                "0.85",
                "--thresholds-json",
                "",
            ],
        },
        {
            "name": "tuned_per_type",
            "per_defect_cmd": [
                str(args.python),
                "scripts/evaluate_per_defect.py",
                "--labels",
                str(labels_path),
                "--splits",
                "test",
                "--out-dir",
                str(out_dir / "tuned_per_type"),
                "--out-md",
                str(out_dir / "tuned_per_type" / "per_defect_report.md"),
                "--out-csv",
                str(out_dir / "tuned_per_type" / "defect_metrics.csv"),
                "--out-csv-by-type",
                str(out_dir / "tuned_per_type" / "defect_metrics_by_type.csv"),
                "--thresholds-json",
                "gdd/models/defect_thresholds.json",
            ],
            "confusion_cmd": [
                str(args.python),
                "scripts/evaluate_defect_confusion.py",
                "--labels",
                str(labels_path),
                "--splits",
                "test",
                "--out-csv",
                str(out_dir / "tuned_per_type" / "defect_confusion_predictions.csv"),
                "--out-dir",
                str(out_dir / "tuned_per_type"),
                "--thresholds-json",
                "gdd/models/defect_thresholds.json",
            ],
        },
    ]

    for mode in modes:
        _run_command(mode["per_defect_cmd"], workdir=repo_root)
        _run_command(mode["confusion_cmd"], workdir=repo_root)

    default_df = _load_metrics_csv(out_dir / "default_ui" / "defect_metrics.csv", "default_ui")
    tuned_df = _load_metrics_csv(out_dir / "tuned_per_type" / "defect_metrics.csv", "tuned_per_type")
    all_labels = pd.DataFrame({"label": list(DEFECT_LABELS)})
    compare_df = all_labels.merge(default_df, on="label", how="left").merge(tuned_df, on="label", how="left")
    compare_df["f1_delta"] = compare_df["tuned_per_type_f1"].fillna(0.0) - compare_df["default_ui_f1"].fillna(0.0)
    compare_df["precision_delta"] = compare_df["tuned_per_type_precision"].fillna(0.0) - compare_df["default_ui_precision"].fillna(0.0)
    compare_df["recall_delta"] = compare_df["tuned_per_type_recall"].fillna(0.0) - compare_df["default_ui_recall"].fillna(0.0)
    compare_path = out_dir / "defect_mode_comparison.csv"
    compare_df.to_csv(compare_path, index=False)

    summary_lines = [
        "# AI-Generated Dual Evaluation Summary",
        "",
        f"- Labels file: `{labels_path}`",
        f"- Manifest audit: `{manifest_audit_path}`",
        f"- Classifier model selected: `{best_name}`",
        f"- Classifier metrics JSON: `{metrics_path}`",
        f"- Classifier predictions CSV: `{pred_path}`",
        f"- Defect comparison CSV: `{compare_path}`",
        "",
        "## Classifier Test Metrics",
        "",
        f"- Accuracy: `{test_metrics['accuracy']:.3f}`",
        f"- Macro precision: `{test_metrics['macro_precision']:.3f}`",
        f"- Macro recall: `{test_metrics['macro_recall']:.3f}`",
        f"- Macro F1: `{test_metrics['macro_f1']:.3f}`",
        "",
        "## Defect Modes",
        "",
        "- `default_ui`: structural `0.65`, surface `0.85`",
        "- `tuned_per_type`: thresholds loaded from `gdd/models/defect_thresholds.json`",
        "",
    ]
    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {manifest_audit_path}")
    print(f"Wrote: {metrics_path}")
    print(f"Wrote: {pred_path}")
    print(f"Wrote: {cm_path}")
    print(f"Wrote: {compare_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
