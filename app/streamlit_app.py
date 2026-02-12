from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gdd.core.image_io import resize_max_side
from gdd.core.pipeline import GDDPipeline
from gdd.core.viz import draw_defects, overlay_mask


st.set_page_config(page_title="GDD — Glove Defect Detection", layout="wide")


def _to_bgr(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image")
    return bgr


def main() -> None:
    st.title("Glove Defect Detection (GDD)")
    st.caption("Classical CV pipeline (no Haar cascade, no TensorFlow, no template matching).")

    pipeline = GDDPipeline.load_default()
    focus_labels = ["missing_finger", "extra_fingers", "hole", "discoloration", "damaged_by_fold"]

    with st.sidebar:
        st.subheader("Display filters")
        min_struct_score = st.slider("Min score (structural)", min_value=0.0, max_value=1.0, value=0.65, step=0.01)
        min_surface_score = st.slider("Min score (surface)", min_value=0.0, max_value=1.0, value=0.85, step=0.01)
        max_boxes = st.slider("Max boxes to draw", min_value=1, max_value=100, value=30, step=1)
        st.subheader("Defect Scope")
        focus_only = st.checkbox("Focus mode (target defects only)", value=True)
        if focus_only:
            selected_labels = focus_labels
            st.caption("Focus labels: missing/extra fingers, hole, discoloration, damaged_by_fold")
        else:
            selected_labels = st.multiselect(
                "Show only selected labels",
                options=[
                    "missing_finger",
                    "extra_fingers",
                    "hole",
                    "discoloration",
                    "damaged_by_fold",
                    "tear",
                    "stain_dirty",
                    "spotting",
                    "plastic_contamination",
                    "wrinkles_dent",
                    "inside_out",
                    "improper_roll",
                    "incomplete_beading",
                ],
                default=focus_labels,
            )

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
            return score >= float(min_struct_score)
        return score >= float(min_surface_score)

    tab1, tab2 = st.tabs(["Single Image", "Batch Folder"])

    with tab1:
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])
        if uploaded is None:
            st.info("Upload a glove image to run inference.")
        else:
            bgr = _to_bgr(uploaded)
            bgr = resize_max_side(bgr, max_side=1200)
            res = pipeline.infer(
                bgr,
                focus_only=bool(focus_only),
                allowed_labels=set(selected_labels) if selected_labels else None,
            )

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Input")
                st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
            with col2:
                st.subheader("Overlay")
                over = overlay_mask(bgr, res.glove_mask)
                draw_list = [d for d in res.defects if d.bbox is not None and passes_threshold(d.label, float(d.score))]
                draw_list.sort(key=lambda d: float(d.score), reverse=True)
                draw_list = draw_list[: int(max_boxes)]
                over = draw_defects(over, draw_list)
                st.image(cv2.cvtColor(over, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")

            st.markdown("### Predictions")
            st.write({"glove_type": res.glove_type, "glove_type_score": round(res.glove_type_score, 3)})
            st.markdown("### Defects")
            if not res.defects:
                st.write("No defects detected (or below thresholds).")
            else:
                show = [d for d in res.defects if passes_threshold(d.label, float(d.score))]
                st.table(
                    [
                        {
                            "label": d.label,
                            "score": round(float(d.score), 3),
                            "bbox": (d.bbox.as_xyxy() if d.bbox else None),
                        }
                        for d in show
                    ]
                )

    with tab2:
        st.markdown("Process a local folder and write overlay images + a CSV under `results/`.")
        folder = st.text_input("Folder path (local)", value="data/raw")
        max_images = st.slider("Max images", min_value=1, max_value=500, value=50, step=1)
        run = st.button("Run batch")
        if run:
            in_dir = Path(folder)
            if not in_dir.exists():
                st.error(f"Folder not found: {in_dir}")
            else:
                paths = []
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                    paths.extend(in_dir.rglob(ext))
                paths = sorted(paths)[: int(max_images)]
                if not paths:
                    st.warning("No images found in that folder.")
                else:
                    out_dir = Path("results/overlays")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    rows = []
                    prog = st.progress(0)
                    for i, p in enumerate(paths):
                        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                        if bgr is None:
                            continue
                        bgr = resize_max_side(bgr, max_side=1200)
                        res = pipeline.infer(
                            bgr,
                            focus_only=bool(focus_only),
                            allowed_labels=set(selected_labels) if selected_labels else None,
                        )
                        over = overlay_mask(bgr, res.glove_mask)
                        draw_list = [d for d in res.defects if d.bbox is not None and passes_threshold(d.label, float(d.score))]
                        draw_list.sort(key=lambda d: float(d.score), reverse=True)
                        draw_list = draw_list[: int(max_boxes)]
                        over = draw_defects(over, draw_list)

                        out_path = out_dir / (p.stem + "_overlay.png")
                        cv2.imwrite(str(out_path), over)

                        rows.append(
                            {
                                "file": str(p),
                                "pred_glove_type": res.glove_type,
                                "pred_glove_type_score": res.glove_type_score,
                                "pred_defects": "|".join([d.label for d in res.defects if passes_threshold(d.label, float(d.score))]),
                            }
                        )
                        prog.progress(int(100 * (i + 1) / len(paths)))

                    csv_path = Path("results/batch_results.csv")
                    import pandas as pd

                    pd.DataFrame(rows).to_csv(csv_path, index=False)
                    st.success(f"Wrote {csv_path} and {out_dir}")


if __name__ == "__main__":
    main()
