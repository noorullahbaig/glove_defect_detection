from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gdd.core.image_io import resize_max_side
from gdd.core.labels import DEFECT_LABELS
from gdd.core.pipeline import GDDPipeline
from gdd.core.viz import draw_defects, overlay_mask


st.set_page_config(page_title="GDD — Glove Defect Detection", layout="wide")


def _to_bgr(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image")
    return bgr


def _to_bgr_bytes(data: bytes) -> np.ndarray:
    buf = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image")
    return bgr


def _put_badge(bgr: np.ndarray, text: str, color: tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    out = bgr.copy()
    cv2.rectangle(out, (10, 10), (10 + 520, 10 + 44), (0, 0, 0), -1)
    cv2.putText(out, text, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return out


@st.cache_resource
def _load_pipeline() -> GDDPipeline:
    return GDDPipeline.load_default()


def main() -> None:
    st.title("Glove Defect Detection (GDD)")
    st.caption("Classical CV pipeline (no Haar cascade, no TensorFlow, no template matching).")

    pipeline = _load_pipeline()

    with st.sidebar:
        st.subheader("Controls")
        min_struct_score = st.slider("Min score (structural)", min_value=0.0, max_value=1.0, value=0.65, step=0.01)
        min_surface_score = st.slider("Min score (surface)", min_value=0.0, max_value=1.0, value=0.85, step=0.01)
        max_boxes = st.slider("Max boxes to draw", min_value=1, max_value=100, value=30, step=1)
        st.subheader("Defect To Analyze")
        defect_options = ["All"] + list(DEFECT_LABELS)
        selected_defect = st.selectbox("Defect", options=defect_options, index=0)

        with st.expander("Defect applicability (scope)"):
            st.markdown(
                """
| Defect | Latex | Leather | Fabric knit |
|---|---:|---:|---:|
| hole, tear | Yes | Yes | Yes |
| discoloration, stain_dirty, spotting | Yes | Yes | Yes |
| plastic_contamination | Yes | Yes | Yes |
| wrinkles_dent, damaged_by_fold, inside_out | Yes | Yes | Yes |
| missing_finger, extra_fingers | Yes | Yes (rare) | Yes (rare) |
| improper_roll, incomplete_beading | Yes | No | No |
                """.strip()
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

    allowed_labels = None if selected_defect == "All" else {str(selected_defect)}

    tab1, tab2 = st.tabs(["Compare Uploads", "Batch Folder"])

    with tab1:
        st.caption("Upload multiple images; prior uploads stay visible for side-by-side comparison.")
        if "gdd_gallery" not in st.session_state:
            st.session_state.gdd_gallery = []  # list[dict{name:str, data:bytes}]

        uploads = st.file_uploader("Upload images", type=["png", "jpg", "jpeg", "bmp"], accept_multiple_files=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            add = st.button("Add uploads")
        with c2:
            clear = st.button("Clear all")
        with c3:
            remove_names = st.multiselect(
                "Remove selected",
                options=[it["name"] for it in st.session_state.gdd_gallery],
                default=[],
            )
            remove = st.button("Remove")

        if clear:
            st.session_state.gdd_gallery = []
        if remove and remove_names:
            st.session_state.gdd_gallery = [it for it in st.session_state.gdd_gallery if it["name"] not in set(remove_names)]
        if add and uploads:
            existing = {(it["name"], len(it["data"])) for it in st.session_state.gdd_gallery}
            for uf in uploads:
                data = uf.getvalue()
                key = (str(uf.name), int(len(data)))
                if key in existing:
                    continue
                st.session_state.gdd_gallery.append({"name": str(uf.name), "data": data})
                existing.add(key)

        if not st.session_state.gdd_gallery:
            st.info("Upload images and click 'Add uploads' to run inference.")
        else:
            rows = []
            cols = st.columns(3)
            for idx, it in enumerate(st.session_state.gdd_gallery):
                name = it["name"]
                bgr = _to_bgr_bytes(it["data"])
                bgr = resize_max_side(bgr, max_side=1200)
                res = pipeline.infer(bgr, focus_only=False, allowed_labels=allowed_labels)

                over = overlay_mask(bgr, res.glove_mask)
                draw_list = [d for d in res.defects if d.bbox is not None and passes_threshold(str(d.label), float(d.score))]
                draw_list.sort(key=lambda d: float(d.score), reverse=True)
                draw_list = draw_list[: int(max_boxes)]
                over = draw_defects(over, draw_list)

                sel_score = None
                sel_present = None
                if selected_defect != "All":
                    scores = [float(d.score) for d in res.defects if str(d.label) == str(selected_defect)]
                    sel_score = float(max(scores)) if scores else 0.0
                    sel_present = bool(passes_threshold(str(selected_defect), float(sel_score)))
                    badge = f"{selected_defect}: {sel_score:.2f} ({'present' if sel_present else 'absent'})"
                    over = _put_badge(over, badge, color=(0, 255, 0) if sel_present else (0, 0, 255))

                pred_defects = [d for d in res.defects if passes_threshold(str(d.label), float(d.score))]
                rows.append(
                    {
                        "file": name,
                        "pred_glove_type": res.glove_type,
                        "glove_type_score": round(float(res.glove_type_score), 3),
                        "selected_defect": (None if selected_defect == "All" else selected_defect),
                        "selected_defect_score": (None if sel_score is None else round(float(sel_score), 3)),
                        "selected_defect_present": (None if sel_present is None else bool(sel_present)),
                        "pred_defects": "|".join(sorted({str(d.label) for d in pred_defects})),
                    }
                )

                with cols[idx % 3]:
                    st.markdown(f"**{name}**")
                    st.caption(f"glove_type={res.glove_type} ({res.glove_type_score:.2f})")
                    st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
                    st.image(cv2.cvtColor(over, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")

            st.markdown("### Comparison Table")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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
                            focus_only=False,
                            allowed_labels=allowed_labels,
                        )
                        over = overlay_mask(bgr, res.glove_mask)
                        draw_list = [d for d in res.defects if d.bbox is not None and passes_threshold(d.label, float(d.score))]
                        draw_list.sort(key=lambda d: float(d.score), reverse=True)
                        draw_list = draw_list[: int(max_boxes)]
                        over = draw_defects(over, draw_list)

                        out_path = out_dir / (p.stem + "_overlay.png")
                        cv2.imwrite(str(out_path), over)

                        sel_score = None
                        if selected_defect != "All":
                            scores = [float(d.score) for d in res.defects if str(d.label) == str(selected_defect)]
                            sel_score = float(max(scores)) if scores else 0.0

                        rows.append(
                            {
                                "file": str(p),
                                "pred_glove_type": res.glove_type,
                                "pred_glove_type_score": res.glove_type_score,
                                "selected_defect": (None if selected_defect == "All" else selected_defect),
                                "selected_defect_score": sel_score,
                                "pred_defects": "|".join([d.label for d in res.defects if passes_threshold(d.label, float(d.score))]),
                            }
                        )
                        prog.progress(int(100 * (i + 1) / len(paths)))

                    csv_path = Path("results/batch_results.csv")
                    pd.DataFrame(rows).to_csv(csv_path, index=False)
                    st.success(f"Wrote {csv_path} and {out_dir}")


if __name__ == "__main__":
    main()
