from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gdd.core.dataset import load_labels_csv, parse_defect_labels, validate_labels_df
from gdd.core.image_io import read_image, resize_max_side
from gdd.core.labels import DEFECT_LABELS
from gdd.core.pipeline import GDDPipeline
from gdd.core.preprocess import preprocess
from gdd.core.segmentation import segment_glove
from gdd.core.defect_detectors import detect_defects


def _f1(tp: int, fp: int, fn: int) -> float:
    p = tp / float(tp + fp + 1e-9)
    r = tp / float(tp + fn + 1e-9)
    return (2.0 * p * r) / float(p + r + 1e-9)


def _fbeta(tp: int, fp: int, fn: int, beta: float = 2.0) -> tuple[float, float, float]:
    p = tp / float(tp + fp + 1e-9)
    r = tp / float(tp + fn + 1e-9)
    b2 = float(beta) * float(beta)
    score = ((1.0 + b2) * p * r) / float((b2 * p) + r + 1e-9)
    return float(score), float(p), float(r)


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
SURFACE_LABELS = {"discoloration", "spotting", "stain_dirty", "plastic_contamination"}


def _default_threshold_for_label(label: str) -> float:
    if label in STRUCTURAL_LABELS:
        if label == "incomplete_beading":
            return 0.80
        if label in {"hole", "tear"}:
            return 0.80
        if label in {"wrinkles_dent", "damaged_by_fold"}:
            return 0.65
        return 0.70
    return 0.85


def _best_threshold_for_label(
    df_scores: pd.DataFrame,
    label: str,
    *,
    beta: float = 2.0,
    recall_weight: float = 0.22,
    fp_penalty: float = 0.05,
    min_precision: float = 0.08,
    thr_low: int = 15,
    thr_high: int = 95,
) -> tuple[float, float]:
    if df_scores.empty:
        return _default_threshold_for_label(label), 0.0
    best_thr = _default_threshold_for_label(label)
    best_f1 = -1.0
    best_obj = -1e9
    y_true = df_scores["true"].astype(bool).tolist()
    y_scores = df_scores["score"].astype(float).tolist()
    lo = int(max(5, min(95, thr_low)))
    hi = int(max(lo, min(95, thr_high)))
    for i in range(lo, hi + 1):
        thr = i / 100.0
        tp = fp = fn = 0
        for truth, score in zip(y_true, y_scores, strict=True):
            pred = bool(score >= thr)
            if truth and pred:
                tp += 1
            elif (not truth) and pred:
                fp += 1
            elif truth and (not pred):
                fn += 1
        f_beta, precision, recall = _fbeta(tp, fp, fn, beta=float(beta))
        n_neg = max(1, len(y_true) - int(sum(1 for x in y_true if x)))
        fp_rate = fp / float(n_neg)
        objective = float(f_beta + float(recall_weight) * recall - float(fp_penalty) * fp_rate)
        if precision < float(min_precision):
            objective -= float(min_precision - precision) * 0.30
        if (objective > best_obj) or (abs(objective - best_obj) <= 1e-9 and thr > best_thr):
            best_obj = objective
            best_f1 = float(_f1(tp, fp, fn))
            best_thr = thr
    return float(best_thr), float(max(0.0, best_f1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/ai_generated/labels.csv", help="Path to labels CSV")
    ap.add_argument(
        "--splits",
        default="all",
        help="Comma-separated splits to tune on (e.g. train,val) or 'all' for all rows",
    )
    ap.add_argument("--out-json", default="gdd/models/defect_thresholds.json", help="Output JSON path")
    ap.add_argument("--out-csv", default="results/tuned_thresholds_metrics.csv", help="Output CSV summary path")
    ap.add_argument("--max-side", type=int, default=900, help="Resize max side before inference")
    args = ap.parse_args()

    df = load_labels_csv(args.labels)
    errs = validate_labels_df(df)
    if errs:
        raise SystemExit("labels.csv validation failed:\n- " + "\n- ".join(errs))

    split_arg = str(args.splits).strip().lower()
    if split_arg == "all":
        tune_df = df.copy()
    else:
        wanted_splits = {s.strip() for s in split_arg.split(",") if s.strip()}
        tune_df = df[df["split"].astype(str).isin(wanted_splits)].copy()
    if tune_df.empty:
        raise SystemExit(f"No rows selected for splits='{args.splits}'.")
    use_val_priority = bool(split_arg != "all")

    pipeline = GDDPipeline.load_default()
    rows: list[dict] = []
    for _, row in tune_df.iterrows():
        path = str(row["file"])
        glove_type = str(row.get("glove_type", "unknown")).strip().lower()
        split = str(row.get("split", ""))
        img = read_image(path).bgr
        img = resize_max_side(img, max_side=int(args.max_side))
        # Use type-specific segmentation + forced defect glove-type to keep this
        # deterministic and fast (avoid auto-seg trial loops during tuning).
        bgr_p = preprocess(img)
        seg_cfg = pipeline.get_profile_seg_cfg(glove_type)
        seg = segment_glove(bgr_p, cfg=seg_cfg)
        defects, _anom = detect_defects(
            bgr_p,
            seg.glove_mask,
            seg.glove_mask_filled,
            glove_type=glove_type,
            focus_only=False,
            allowed_labels=None,
        )
        truth = set(parse_defect_labels(str(row.get("defect_labels", ""))))

        score_map = {label: 0.0 for label in DEFECT_LABELS}
        for d in defects:
            lab = str(d.label)
            if lab not in score_map:
                continue
            score_map[lab] = max(float(score_map[lab]), float(d.score))

        for label in DEFECT_LABELS:
            rows.append(
                {
                    "file": path,
                    "split": split,
                    "glove_type": glove_type,
                    "label": label,
                    "true": bool(label in truth),
                    "score": float(score_map.get(label, 0.0)),
                }
            )

    sdf = pd.DataFrame(rows)
    if sdf.empty:
        raise SystemExit("No scores generated from validation split.")

    per_type_thresholds: dict[str, dict[str, float]] = {}
    summary_rows: list[dict] = []
    for gt in ("latex", "leather", "fabric"):
        gt_df = sdf[sdf["glove_type"].astype(str) == gt].copy()
        if gt_df.empty:
            continue
        per_type_thresholds[gt] = {}
        for label in DEFECT_LABELS:
            lab_all = gt_df[gt_df["label"].astype(str) == label].copy()
            lab_val = lab_all[lab_all["split"].astype(str) == "val"].copy() if use_val_priority else lab_all.iloc[0:0].copy()
            n_pos_val = int(lab_val["true"].astype(int).sum()) if not lab_val.empty else 0
            n_pos_trainval = int(lab_all["true"].astype(int).sum()) if not lab_all.empty else 0

            source = "default"
            is_struct = bool(label in STRUCTURAL_LABELS)
            beta = 2.4 if is_struct else 2.2
            recall_weight = 0.26 if is_struct else 0.22
            fp_pen = 0.045 if is_struct else 0.055
            min_p = 0.10 if is_struct else 0.08
            thr_lo = 15
            if n_pos_val >= 2:
                source = "val"
                thr, f1 = _best_threshold_for_label(
                    lab_val,
                    label,
                    beta=beta,
                    recall_weight=recall_weight,
                    fp_penalty=fp_pen,
                    min_precision=min_p,
                    thr_low=thr_lo,
                    thr_high=95,
                )
            elif n_pos_trainval >= 2:
                source = "train_val"
                thr, f1 = _best_threshold_for_label(
                    lab_all,
                    label,
                    beta=beta,
                    recall_weight=recall_weight,
                    fp_penalty=fp_pen + 0.02,
                    min_precision=min_p,
                    thr_low=thr_lo,
                    thr_high=95,
                )
                d = float(_default_threshold_for_label(label))
                if is_struct:
                    thr = min(d + 0.08, max(d - 0.28, float(thr)))
                else:
                    thr = min(d + 0.05, max(d - 0.25, float(thr)))
            elif n_pos_trainval >= 1:
                source = "train_val_lowpos"
                thr, f1 = _best_threshold_for_label(
                    lab_all,
                    label,
                    beta=beta,
                    recall_weight=recall_weight + 0.02,
                    fp_penalty=fp_pen + 0.03,
                    min_precision=min_p,
                    thr_low=thr_lo,
                    thr_high=95,
                )
                d = float(_default_threshold_for_label(label))
                if is_struct:
                    thr = min(d + 0.06, max(d - 0.30, float(thr)))
                else:
                    thr = min(d + 0.04, max(d - 0.28, float(thr)))
                if float(f1) <= 0.0:
                    thr = max(0.15, d - (0.18 if is_struct else 0.22))
            else:
                thr = _default_threshold_for_label(label)
                f1 = 0.0

            per_type_thresholds[gt][label] = float(thr)
            summary_rows.append(
                {
                    "glove_type": gt,
                    "label": label,
                    "best_threshold": float(thr),
                    "best_f1_on_val": float(f1),
                    "n_pos_val": int(n_pos_val),
                    "n_pos_train_val": int(n_pos_trainval),
                    "n_rows_val": int(len(lab_val)),
                    "n_rows_train_val": int(len(lab_all)),
                    "threshold_source": source,
                }
            )

    out_json = Path(str(args.out_json))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "source_labels": str(args.labels),
        "split_used": f"adaptive_{args.splits}",
        "per_type": per_type_thresholds,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    out_csv = Path(str(args.out_csv))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
