from __future__ import annotations

import json
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
from gdd.core.preprocess import preprocess
from gdd.core.segmentation import SegmentationConfig, segment_glove_debug
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


@st.cache_data(show_spinner=False)
def _load_tuned_thresholds(path: str = "gdd/models/defect_thresholds.json") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _apply_seg_preset(preset: str) -> None:
    # These set the widget state keys used in the sidebar.
    if preset == "Balanced":
        st.session_state.update(
            {
                "seg_force_candidate": "auto",
                "seg_force_refine": "auto",
                "seg_cap_bt": 0.10,
                "seg_open_k": 3,
                "seg_close_k": 7,
                "seg_edge_kd": 1.0,
                "seg_edge_ring": 1.0,
                "seg_edge_area": 1.22,
                "seg_upright": True,
                "seg_roi_frac": 0.60,
                "seg_upright_kd": 1.8,
                "seg_upright_ring": 1.55,
                "seg_upright_area": 1.30,
                "seg_valley": True,
                "seg_valley_prom": 0.05,
                "seg_notch_w": 0.07,
                "seg_notch_d": 0.22,
                "seg_notch_cap": 0.42,
                "seg_bg_pct": 96.0,
                "seg_bg_margin": 6.0,
                "seg_thin_dt": 0.025,
                "seg_shadow_strict": 1.0,
                "seg_webbing_on": True,
                "seg_webbing_trig": 0.12,
                "seg_webbing_erode": 0.018,
                "seg_webbing_overlap": 0.18,
                "seg_reach_carve_on": True,
                "seg_reach_carve_strength": 1.0,
                "seg_profile_name": "balanced",
                "seg_gap_mode": "hybrid",
                "seg_chroma_min_area": 0.20,
                "seg_chroma_min_iou": 0.25,
                "seg_boundary_prune_on": True,
                "seg_boundary_prune_edge_rescue": True,
                "seg_boundary_prune_skip_low_leak": True,
                "seg_gap_geom_solidity": 0.90,
                "seg_gap_max_rm": 0.06,
            }
        )
    elif preset == "Preserve Edges":
        st.session_state.update(
            {
                "seg_force_candidate": "auto",
                "seg_force_refine": "auto",
                "seg_cap_bt": 0.12,
                "seg_open_k": 2,
                "seg_close_k": 5,
                "seg_edge_kd": 1.35,
                "seg_edge_ring": 1.25,
                "seg_edge_area": 1.30,
                "seg_upright": True,
                "seg_roi_frac": 0.62,
                "seg_upright_kd": 2.1,
                "seg_upright_ring": 1.75,
                "seg_upright_area": 1.35,
                "seg_valley": True,
                "seg_valley_prom": 0.045,
                "seg_notch_w": 0.08,
                "seg_notch_d": 0.25,
                "seg_notch_cap": 0.46,
                "seg_bg_pct": 97.0,
                "seg_bg_margin": 7.0,
                "seg_thin_dt": 0.030,
                "seg_shadow_strict": 0.9,
                "seg_webbing_on": True,
                "seg_webbing_trig": 0.10,
                "seg_webbing_erode": 0.016,
                "seg_webbing_overlap": 0.18,
                "seg_reach_carve_on": True,
                "seg_reach_carve_strength": 0.7,
                "seg_profile_name": "balanced",
                "seg_gap_mode": "hybrid",
                "seg_chroma_min_area": 0.20,
                "seg_chroma_min_iou": 0.25,
                "seg_boundary_prune_on": True,
                "seg_boundary_prune_edge_rescue": True,
                "seg_boundary_prune_skip_low_leak": True,
                "seg_gap_geom_solidity": 0.90,
                "seg_gap_max_rm": 0.05,
            }
        )
    elif preset == "Strict Gaps":
        st.session_state.update(
            {
                "seg_force_candidate": "auto",
                "seg_force_refine": "auto",
                "seg_cap_bt": 0.10,
                "seg_open_k": 3,
                "seg_close_k": 7,
                "seg_edge_kd": 1.10,
                "seg_edge_ring": 1.05,
                "seg_edge_area": 1.24,
                "seg_upright": True,
                "seg_roi_frac": 0.60,
                "seg_upright_kd": 1.9,
                "seg_upright_ring": 1.6,
                "seg_upright_area": 1.30,
                "seg_valley": True,
                "seg_valley_prom": 0.040,
                "seg_notch_w": 0.09,
                "seg_notch_d": 0.28,
                "seg_notch_cap": 0.50,
                "seg_bg_pct": 95.0,
                "seg_bg_margin": 4.0,
                "seg_thin_dt": 0.020,
                "seg_shadow_strict": 0.8,
                "seg_webbing_on": True,
                "seg_webbing_trig": 0.08,
                "seg_webbing_erode": 0.020,
                "seg_webbing_overlap": 0.15,
                "seg_reach_carve_on": True,
                "seg_reach_carve_strength": 1.6,
                "seg_profile_name": "balanced",
                "seg_gap_mode": "hybrid",
                "seg_chroma_min_area": 0.20,
                "seg_chroma_min_iou": 0.25,
                "seg_boundary_prune_on": True,
                "seg_boundary_prune_edge_rescue": True,
                "seg_boundary_prune_skip_low_leak": True,
                "seg_gap_geom_solidity": 0.88,
                "seg_gap_max_rm": 0.08,
            }
        )


def _seg_cfg_from_sidebar(enabled: bool) -> SegmentationConfig | None:
    if not enabled:
        return None

    force_candidate = st.session_state.get("seg_force_candidate", "auto")
    force_candidate = None if force_candidate == "auto" else str(force_candidate)
    force_refine = st.session_state.get("seg_force_refine", "auto")
    force_refine = None if force_refine == "auto" else str(force_refine)

    return SegmentationConfig(
        force_candidate=force_candidate,
        force_refine=force_refine,
        profile_name=str(st.session_state.get("seg_profile_name", "balanced")),
        gap_carve_mode=str(st.session_state.get("seg_gap_mode", "hybrid")),
        chroma_min_area_frac=float(st.session_state.get("seg_chroma_min_area", 0.20)),
        chroma_min_iou_silhouette=float(st.session_state.get("seg_chroma_min_iou", 0.25)),
        grabcut_cap_border_touch=float(st.session_state.get("seg_cap_bt", 0.10)),
        grabcut_open_k=int(st.session_state.get("seg_open_k", 3)),
        grabcut_close_k=int(st.session_state.get("seg_close_k", 7)),
        edge_recover_enabled=True,
        edge_recover_kd_scale=float(st.session_state.get("seg_edge_kd", 1.0)),
        edge_recover_ring_scale=float(st.session_state.get("seg_edge_ring", 1.0)),
        edge_recover_max_area_mul=float(st.session_state.get("seg_edge_area", 1.22)),
        upright_enabled=bool(st.session_state.get("seg_upright", True)),
        upright_roi_frac=float(st.session_state.get("seg_roi_frac", 0.60)),
        upright_kd_scale=float(st.session_state.get("seg_upright_kd", 1.8)),
        upright_ring_scale=float(st.session_state.get("seg_upright_ring", 1.55)),
        upright_max_area_mul=float(st.session_state.get("seg_upright_area", 1.30)),
        valley_carve_enabled=bool(st.session_state.get("seg_valley", True)),
        valley_prom_frac=float(st.session_state.get("seg_valley_prom", 0.05)),
        valley_notch_w_frac=float(st.session_state.get("seg_notch_w", 0.07)),
        valley_notch_depth_frac=float(st.session_state.get("seg_notch_d", 0.22)),
        valley_notch_depth_cap_frac=float(st.session_state.get("seg_notch_cap", 0.42)),
        bg_like_percentile=float(st.session_state.get("seg_bg_pct", 96.0)),
        bg_like_margin=float(st.session_state.get("seg_bg_margin", 6.0)),
        thin_dt_frac=float(st.session_state.get("seg_thin_dt", 0.025)),
        shadow_strictness=float(st.session_state.get("seg_shadow_strict", 1.0)),
        webbing_restore_enabled=bool(st.session_state.get("seg_webbing_on", True)),
        webbing_trigger_ratio=float(st.session_state.get("seg_webbing_trig", 0.12)),
        webbing_erode_k_frac=float(st.session_state.get("seg_webbing_erode", 0.018)),
        webbing_convexity_enabled=bool(st.session_state.get("seg_webbing_convex", True)),
        webbing_solidity_thr=float(st.session_state.get("seg_webbing_solidity", 0.92)),
        webbing_depth_frac=float(st.session_state.get("seg_webbing_depth", 0.030)),
        webbing_wedge_dilate_px=int(st.session_state.get("seg_webbing_wedge", 7)),
        webbing_reach_overlap_thr=float(st.session_state.get("seg_webbing_overlap", 0.18)),
        reach_carve_enabled=bool(st.session_state.get("seg_reach_carve_on", True)),
        reach_carve_strength=float(st.session_state.get("seg_reach_carve_strength", 1.0)),
        boundary_prune_enabled=bool(st.session_state.get("seg_boundary_prune_on", True)),
        boundary_prune_edge_rescue=bool(st.session_state.get("seg_boundary_prune_edge_rescue", True)),
        boundary_prune_skip_if_low_leak=bool(st.session_state.get("seg_boundary_prune_skip_low_leak", True)),
        gap_geometry_fallback_solidity=float(st.session_state.get("seg_gap_geom_solidity", 0.90)),
        gap_max_remove_frac=float(st.session_state.get("seg_gap_max_rm", 0.06)),
        # Type-specific strategy flags.
        latex_edge_multiscale=bool(st.session_state.get("seg_latex_edge_ms", False)),
        leather_saturation_candidate=bool(st.session_state.get("seg_leather_sat", False)),
        fabric_variance_candidate=bool(st.session_state.get("seg_fabric_var", False)),
        grabcut_trimap_enabled=bool(st.session_state.get("seg_gc_trimap", False)),
        grabcut_trimap_only_if_low_confidence=bool(st.session_state.get("seg_gc_trimap_lowconf", True)),
        chan_vese_enabled=bool(st.session_state.get("seg_chan_vese", False)),
        chan_vese_only_if_low_confidence=bool(st.session_state.get("seg_chan_vese_lowconf", True)),
        chan_vese_iters=int(st.session_state.get("seg_chan_vese_iters", 80)),
        chan_vese_max_side=int(st.session_state.get("seg_chan_vese_side", 512)),
        shape_plausibility_enabled=bool(st.session_state.get("seg_shape_plaus", True)),
        shape_plausibility_weight=float(st.session_state.get("seg_shape_weight", 0.08)),
        # Edge-based finger separation + GrabCut seeding (always enabled).
        edge_finger_separation_enabled=True,
        grabcut_erode_frac=float(st.session_state.get("seg_gc_erode_frac", 0.008)),
        grabcut_dilate_frac=float(st.session_state.get("seg_gc_dilate_frac", 0.030)),
    )


def main() -> None:
    st.title("Glove Defect Detection (GDD)")

    pipeline = _load_pipeline()
    tuned_thresholds = _load_tuned_thresholds()
    has_tuned_thresholds = bool(isinstance(tuned_thresholds.get("per_type"), dict) and tuned_thresholds.get("per_type"))

    if pipeline.glove_type_model is None:
        warn = (
            "Glove-type classifier is unavailable. Predictions will show `unknown` until the trained model "
            f"is available at `{pipeline.glove_type_model_path}`."
        )
        if pipeline.glove_type_model_error:
            warn += f" Load status: {pipeline.glove_type_model_error}"
        st.warning(warn)

    with st.sidebar:
        st.subheader("Controls")
        min_struct_score = st.slider("Min score (structural)", min_value=0.0, max_value=1.0, value=0.65, step=0.01)
        min_surface_score = st.slider("Min score (surface)", min_value=0.0, max_value=1.0, value=0.85, step=0.01)
        max_boxes = st.slider("Max boxes to draw", min_value=1, max_value=100, value=30, step=1)
        show_seg_debug = st.checkbox("Show segmentation debug", value=True)

        st.subheader("Segmentation Tuning")
        seg_mode = st.selectbox("Segmentation mode", options=["Auto (type-conditional)", "Manual (sliders)"], index=0)
        seg_tuning = bool(seg_mode == "Manual (sliders)")
        preset = st.selectbox("Preset", options=["Balanced", "Preserve Edges", "Strict Gaps"], index=0, disabled=not seg_tuning)
        cpa, cpb = st.columns(2)
        with cpa:
            if st.button("Apply preset", disabled=not seg_tuning):
                _apply_seg_preset(str(preset))
                st.rerun()
        with cpb:
            st.caption("Use the Tuner tab for 1-image tuning.")

        with st.expander("Advanced segmentation sliders", expanded=False):
            st.selectbox(
                "Force candidate",
                options=["auto", "bg_dist", "labL_hi", "labL_lo", "chroma_fg", "texture_fg", "kmeans", "edge_closed", "edge_closed_strong"],
                key="seg_force_candidate",
                disabled=not seg_tuning,
            )
            st.selectbox(
                "Force refine",
                options=["auto", "grabcut", "watershed"],
                key="seg_force_refine",
                disabled=not seg_tuning,
            )
            st.selectbox(
                "Manual profile hint",
                options=["balanced", "latex", "leather", "fabric"],
                key="seg_profile_name",
                disabled=not seg_tuning,
            )
            st.selectbox(
                "Gap carve mode",
                options=["hybrid", "evidence", "geometry"],
                key="seg_gap_mode",
                disabled=not seg_tuning,
            )
            st.slider("Chroma min area frac", min_value=0.05, max_value=0.45, value=0.20, step=0.01, key="seg_chroma_min_area", disabled=not seg_tuning)
            st.slider("Chroma min IoU silhouette", min_value=0.05, max_value=0.60, value=0.25, step=0.01, key="seg_chroma_min_iou", disabled=not seg_tuning)
            st.slider("GrabCut cap border-touch", min_value=0.02, max_value=0.30, value=0.10, step=0.01, key="seg_cap_bt", disabled=not seg_tuning)
            st.slider("GrabCut opening kernel (px)", min_value=1, max_value=11, value=3, step=1, key="seg_open_k", disabled=not seg_tuning)
            st.slider("GrabCut closing kernel (px)", min_value=1, max_value=15, value=7, step=1, key="seg_close_k", disabled=not seg_tuning)
            st.markdown("**GrabCut tri-map seeding**")
            st.checkbox("Enable tri-map (multi-level Otsu)", value=False, key="seg_gc_trimap", disabled=not seg_tuning)
            st.checkbox("Tri-map only if low confidence", value=True, key="seg_gc_trimap_lowconf", disabled=not seg_tuning)

            st.markdown("**Region-based fallback (Chan–Vese)**")
            st.checkbox("Enable Chan–Vese candidate", value=False, key="seg_chan_vese", disabled=not seg_tuning)
            st.checkbox("Chan–Vese only if low confidence", value=True, key="seg_chan_vese_lowconf", disabled=not seg_tuning)
            st.slider("Chan–Vese iters", min_value=10, max_value=200, value=80, step=10, key="seg_chan_vese_iters", disabled=not seg_tuning)
            st.slider("Chan–Vese max side", min_value=160, max_value=900, value=512, step=32, key="seg_chan_vese_side", disabled=not seg_tuning)

            st.markdown("**Wrong-object guard (shape plausibility)**")
            st.checkbox("Enable shape plausibility", value=True, key="seg_shape_plaus", disabled=not seg_tuning)
            st.slider("Shape weight", min_value=0.0, max_value=0.25, value=0.08, step=0.01, key="seg_shape_weight", disabled=not seg_tuning)
            st.checkbox("Boundary prune enabled", value=True, key="seg_boundary_prune_on", disabled=not seg_tuning)
            st.checkbox("Boundary prune edge rescue", value=True, key="seg_boundary_prune_edge_rescue", disabled=not seg_tuning)
            st.checkbox("Boundary prune skip low-leak latex", value=True, key="seg_boundary_prune_skip_low_leak", disabled=not seg_tuning)

            st.markdown("**Edge recovery (global)**")
            st.slider("Edge recover kd scale", min_value=0.0, max_value=2.5, value=1.0, step=0.05, key="seg_edge_kd", disabled=not seg_tuning)
            st.slider("Edge recover ring scale", min_value=0.0, max_value=2.5, value=1.0, step=0.05, key="seg_edge_ring", disabled=not seg_tuning)
            st.slider("Edge recover max area x", min_value=1.00, max_value=1.60, value=1.22, step=0.01, key="seg_edge_area", disabled=not seg_tuning)

            st.markdown("**Upright finger refine**")
            st.checkbox("Enable upright refine", value=True, key="seg_upright", disabled=not seg_tuning)
            st.slider("Finger ROI fraction", min_value=0.45, max_value=0.75, value=0.60, step=0.01, key="seg_roi_frac", disabled=not seg_tuning)
            st.slider("Upright kd scale", min_value=0.0, max_value=3.0, value=1.8, step=0.05, key="seg_upright_kd", disabled=not seg_tuning)
            st.slider("Upright ring scale", min_value=0.0, max_value=3.0, value=1.55, step=0.05, key="seg_upright_ring", disabled=not seg_tuning)
            st.slider("Upright max area x", min_value=1.00, max_value=1.80, value=1.30, step=0.01, key="seg_upright_area", disabled=not seg_tuning)

            st.markdown("**Webbing / gap carve**")
            st.checkbox("Enable valley carve", value=True, key="seg_valley", disabled=not seg_tuning)
            st.slider("Valley prominence (frac height)", min_value=0.01, max_value=0.12, value=0.05, step=0.005, key="seg_valley_prom", disabled=not seg_tuning)
            st.slider("Notch width (frac width)", min_value=0.03, max_value=0.15, value=0.07, step=0.005, key="seg_notch_w", disabled=not seg_tuning)
            st.slider("Notch depth (frac height)", min_value=0.08, max_value=0.45, value=0.22, step=0.01, key="seg_notch_d", disabled=not seg_tuning)
            st.slider("Notch depth cap (frac height)", min_value=0.15, max_value=0.70, value=0.42, step=0.01, key="seg_notch_cap", disabled=not seg_tuning)
            st.slider("Background-like percentile", min_value=80.0, max_value=99.5, value=96.0, step=0.5, key="seg_bg_pct", disabled=not seg_tuning)
            st.slider("Background-like margin", min_value=0.0, max_value=18.0, value=6.0, step=0.5, key="seg_bg_margin", disabled=not seg_tuning)
            st.slider("Thin DT threshold (frac)", min_value=0.005, max_value=0.08, value=0.025, step=0.001, key="seg_thin_dt", disabled=not seg_tuning)
            st.markdown("**Shadow + thumb webbing**")
            st.slider("Shadow strictness", min_value=0.5, max_value=2.5, value=1.0, step=0.05, key="seg_shadow_strict", disabled=not seg_tuning)
            st.checkbox("Enable webbing restore", value=True, key="seg_webbing_on", disabled=not seg_tuning)
            st.slider("Webbing trigger ratio", min_value=0.02, max_value=0.30, value=0.12, step=0.01, key="seg_webbing_trig", disabled=not seg_tuning)
            st.slider("Webbing erosion (frac)", min_value=0.006, max_value=0.050, value=0.018, step=0.001, key="seg_webbing_erode", disabled=not seg_tuning)
            st.checkbox("Enable convexity webbing carve", value=True, key="seg_webbing_convex", disabled=not seg_tuning)
            st.slider("Convexity solidity threshold", min_value=0.80, max_value=0.99, value=0.92, step=0.01, key="seg_webbing_solidity", disabled=not seg_tuning)
            st.slider("Convexity min depth (frac)", min_value=0.010, max_value=0.080, value=0.030, step=0.002, key="seg_webbing_depth", disabled=not seg_tuning)
            st.slider("Convexity wedge dilate (px)", min_value=0, max_value=25, value=7, step=1, key="seg_webbing_wedge", disabled=not seg_tuning)
            st.slider("Convexity bg-overlap threshold", min_value=0.05, max_value=0.55, value=0.18, step=0.01, key="seg_webbing_overlap", disabled=not seg_tuning)
            st.slider("Geometry fallback solidity", min_value=0.80, max_value=0.98, value=0.90, step=0.01, key="seg_gap_geom_solidity", disabled=not seg_tuning)
            st.slider("Gap max remove frac", min_value=0.01, max_value=0.20, value=0.06, step=0.005, key="seg_gap_max_rm", disabled=not seg_tuning)
            st.markdown("**Edge-reachable carve (shadows + gaps)**")
            st.checkbox("Enable edge-reachable carve", value=True, key="seg_reach_carve_on", disabled=not seg_tuning)
            st.slider("Edge-reachable carve strength", min_value=0.0, max_value=2.5, value=1.0, step=0.05, key="seg_reach_carve_strength", disabled=not seg_tuning)
            st.markdown("**Type-specific strategy**")
            st.checkbox("Latex: multi-scale edge candidate", value=False, key="seg_latex_edge_ms", disabled=not seg_tuning)
            st.checkbox("Leather: saturation candidate", value=False, key="seg_leather_sat", disabled=not seg_tuning)
            st.checkbox("Fabric: variance candidate", value=False, key="seg_fabric_var", disabled=not seg_tuning)

        seg_cfg = _seg_cfg_from_sidebar(bool(seg_tuning))
        if not seg_tuning:
            seg_cfg = None
        force_defect_glove_type = st.selectbox("Force defect glove type", options=["Auto", "latex", "leather", "fabric"], index=0)
        threshold_mode_options = ["Manual sliders"]
        if has_tuned_thresholds:
            threshold_mode_options.append("Tuned per type")
        threshold_mode = st.selectbox("Threshold mode", options=threshold_mode_options, index=0)
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
| improper_roll, incomplete_beading | Yes | Yes | Yes |
                """.strip()
            )

    structural_labels = {
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

    def _threshold_for(label: str, glove_type_value: str) -> float:
        if threshold_mode != "Tuned per type":
            if label in structural_labels:
                return float(min_struct_score)
            return float(min_surface_score)
        per_type = tuned_thresholds.get("per_type", {}) if isinstance(tuned_thresholds, dict) else {}
        gt = str(glove_type_value).strip().lower()
        if isinstance(per_type, dict):
            gt_map = per_type.get(gt, {})
            if isinstance(gt_map, dict) and label in gt_map:
                try:
                    return float(gt_map[label])
                except Exception:
                    pass
        if label in structural_labels:
            return float(min_struct_score)
        return float(min_surface_score)

    def passes_threshold(label: str, score: float, glove_type_value: str) -> bool:
        return float(score) >= float(_threshold_for(label, glove_type_value))

    def _defect_eval_glove_type(defect, fallback_glove_type: str) -> str:
        if defect is None:
            return str(fallback_glove_type)
        label = str(getattr(defect, "label", "") or "")
        meta = getattr(defect, "meta", None) or {}
        if label == "improper_roll":
            rescue_profile = str(meta.get("rescue_profile", "")).strip().lower()
            if rescue_profile in {"latex", "fabric", "leather"}:
                return rescue_profile
        return str(fallback_glove_type)

    allowed_labels = None if selected_defect == "All" else {str(selected_defect)}
    forced_glove_type = None if force_defect_glove_type == "Auto" else str(force_defect_glove_type)

    tab1, tab2, tab3 = st.tabs(["Compare Uploads", "Batch Folder", "Segmentation Tuner"])

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
                res = pipeline.infer(
                    bgr,
                    focus_only=False,
                    allowed_labels=allowed_labels,
                    seg_cfg=seg_cfg,
                    force_glove_type=forced_glove_type,
                )
                eval_glove_type = str(forced_glove_type or res.glove_type)
                seg_auto_dbg = dict(pipeline.last_seg_debug_info or {})

                over = overlay_mask(bgr, res.glove_mask)
                draw_list = [
                    d
                    for d in res.defects
                    if d.bbox is not None
                    and passes_threshold(str(d.label), float(d.score), _defect_eval_glove_type(d, eval_glove_type))
                ]
                draw_list.sort(key=lambda d: float(d.score), reverse=True)
                draw_list = draw_list[: int(max_boxes)]
                over = draw_defects(over, draw_list)

                sel_score = None
                sel_present = None
                if selected_defect != "All":
                    sel_defects = [d for d in res.defects if str(d.label) == str(selected_defect)]
                    sel_best = max(sel_defects, key=lambda d: float(d.score)) if sel_defects else None
                    sel_score = float(sel_best.score) if sel_best is not None else 0.0
                    sel_eval_glove_type = _defect_eval_glove_type(sel_best, eval_glove_type)
                    sel_present = bool(passes_threshold(str(selected_defect), float(sel_score), sel_eval_glove_type))
                    badge = f"{selected_defect}: {sel_score:.2f} ({'present' if sel_present else 'absent'})"
                    over = _put_badge(over, badge, color=(0, 255, 0) if sel_present else (0, 0, 255))

                pred_defects = [
                    d
                    for d in res.defects
                    if passes_threshold(str(d.label), float(d.score), _defect_eval_glove_type(d, eval_glove_type))
                ]
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
                    if show_seg_debug:
                        with st.expander("Segmentation debug", expanded=True):
                            bgr_p = preprocess(bgr)
                            dbg_cfg = seg_cfg
                            chosen_profile = str(seg_auto_dbg.get("chosen_profile", ""))
                            if dbg_cfg is None and chosen_profile in {"latex", "leather", "fabric"}:
                                dbg_cfg = pipeline.get_profile_seg_cfg(chosen_profile)
                            seg_res, seg_dbg = segment_glove_debug(bgr_p, cfg=dbg_cfg)
                            if seg_auto_dbg:
                                st.markdown("**Auto profile selection**")
                                mode = str(seg_auto_dbg.get("mode", ""))
                                top_probe_label = str(seg_auto_dbg.get("top_probe_label", ""))
                                top_probe_score = float(seg_auto_dbg.get("top_probe_score", 0.0))
                                if mode == "auto_type_conditional":
                                    st.caption(
                                        f"profile={seg_auto_dbg.get('chosen_profile', 'balanced')} | "
                                        f"top_probe={top_probe_label} ({top_probe_score:.2f})"
                                    )
                                elif mode == "manual_cfg":
                                    st.caption("profile=manual sliders")
                                trials = seg_auto_dbg.get("tried_profiles")
                                if isinstance(trials, list) and trials:
                                    st.dataframe(pd.DataFrame(trials), use_container_width=True, hide_index=True)
                            st.caption(
                                f"chosen={seg_dbg.chosen_method} | bg_white={seg_dbg.bg_is_white} bg_uniform={seg_dbg.bg_is_uniform}"
                            )

                            # ── Type strategy badge ─────────────────────
                            _strategy_map = {
                                "latex": "🧤 Edge-First (Latex)",
                                "leather": "🧤 Chroma-Saturation (Leather)",
                                "fabric": "🧤 Texture-Variance (Fabric)",
                                "balanced": "🧤 Balanced (Auto)",
                            }
                            _profile = str(seg_auto_dbg.get("chosen_profile", "balanced"))
                            _strat_label = _strategy_map.get(_profile, f"🧤 {_profile}")
                            st.info(f"**Strategy**: {_strat_label}")
                            if seg_dbg.scored:
                                st.dataframe(pd.DataFrame(seg_dbg.scored).head(6), use_container_width=True, hide_index=True)
                            if seg_dbg.candidate_validity:
                                rows_valid = [{"method": k, **v} for k, v in seg_dbg.candidate_validity.items()]
                                st.markdown("**Candidate validity**")
                                st.dataframe(pd.DataFrame(rows_valid), use_container_width=True, hide_index=True)
                            st.markdown("**Refine diagnostics**")
                            st.dataframe(
                                pd.DataFrame(
                                    [
                                        {"kind": "carve", **seg_dbg.carve_stats},
                                        {"kind": "prune", **seg_dbg.prune_stats},
                                    ]
                                ),
                                use_container_width=True,
                                hide_index=True,
                            )

                            heat = cv2.applyColorMap(seg_dbg.d_bg_u8, cv2.COLORMAP_TURBO)
                            edges_rgb = cv2.cvtColor(seg_dbg.edges_u8, cv2.COLOR_GRAY2RGB)
                            cdbg1, cdbg2 = st.columns(2)
                            with cdbg1:
                                st.markdown("**Background distance (d_bg)**")
                                st.image(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
                            with cdbg2:
                                st.markdown("**Edges**")
                                st.image(edges_rgb, channels="RGB", width="stretch")

                            top_methods = [d.get("method") for d in (seg_dbg.scored[:2] if seg_dbg.scored else [])]
                            for m in [t for t in top_methods if isinstance(t, str)]:
                                cand = seg_dbg.candidates.get(m)
                                if cand is None:
                                    continue
                                cand_u8 = (cand.astype(np.uint8) * 255).astype(np.uint8)
                                cand_over = overlay_mask(bgr, cand_u8)
                                st.markdown(f"**Candidate: `{m}`**")
                                st.image(cv2.cvtColor(cand_over, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")

                            st.markdown("**Final mask (this run)**")
                            final_over = overlay_mask(bgr, seg_res.glove_mask)
                            st.image(cv2.cvtColor(final_over, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")

                            # ── Shadow mask visualization ───────────────
                            try:
                                from gdd.core.segmentation import _compute_signals, _shadow_like_mask
                                _sig = _compute_signals(bgr_p)
                                _shadow = _shadow_like_mask(_sig, strictness=float(dbg_cfg.shadow_strictness) if dbg_cfg else 1.0)
                                shadow_viz = bgr_p.copy()
                                shadow_viz[_shadow > 0] = (
                                    (shadow_viz[_shadow > 0].astype(np.float32) * 0.4
                                     + np.array([200, 100, 50], dtype=np.float32) * 0.6)
                                ).astype(np.uint8)
                                st.markdown("**Shadow mask**")
                                st.image(cv2.cvtColor(shadow_viz, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
                            except Exception:
                                pass  # graceful fallback if internal import fails

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
                            seg_cfg=seg_cfg,
                            force_glove_type=forced_glove_type,
                        )
                        eval_glove_type = str(forced_glove_type or res.glove_type)
                        over = overlay_mask(bgr, res.glove_mask)
                        draw_list = [
                            d
                            for d in res.defects
                            if d.bbox is not None
                            and passes_threshold(d.label, float(d.score), _defect_eval_glove_type(d, eval_glove_type))
                        ]
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
                                "pred_defects": "|".join(
                                    [
                                        d.label
                                        for d in res.defects
                                        if passes_threshold(d.label, float(d.score), _defect_eval_glove_type(d, eval_glove_type))
                                    ]
                                ),
                            }
                        )
                        prog.progress(int(100 * (i + 1) / len(paths)))

                    csv_path = Path("results/batch_results.csv")
                    pd.DataFrame(rows).to_csv(csv_path, index=False)
                    st.success(f"Wrote {csv_path} and {out_dir}")

    with tab3:
        st.caption("Tune segmentation on a single image with the sidebar sliders, then switch back to Compare Uploads.")
        if not st.session_state.get("gdd_gallery"):
            st.info("Upload at least one image in 'Compare Uploads' to tune it here.")
        else:
            names = [it["name"] for it in st.session_state.gdd_gallery]
            chosen = st.selectbox("Choose an uploaded image", options=names, index=0)
            it = next((x for x in st.session_state.gdd_gallery if x["name"] == chosen), None)
            if it is None:
                st.error("Could not find that image in session.")
            else:
                bgr = _to_bgr_bytes(it["data"])
                bgr = resize_max_side(bgr, max_side=1400)
                bgr_p = preprocess(bgr)

                res = pipeline.infer(
                    bgr,
                    focus_only=False,
                    allowed_labels=allowed_labels,
                    seg_cfg=seg_cfg,
                    force_glove_type=forced_glove_type,
                )
                eval_glove_type = str(forced_glove_type or res.glove_type)
                seg_auto_dbg = dict(pipeline.last_seg_debug_info or {})
                dbg_cfg = seg_cfg
                chosen_profile = str(seg_auto_dbg.get("chosen_profile", ""))
                if dbg_cfg is None and chosen_profile in {"latex", "leather", "fabric"}:
                    dbg_cfg = pipeline.get_profile_seg_cfg(chosen_profile)
                seg_res, seg_dbg = segment_glove_debug(bgr_p, cfg=dbg_cfg)

                cA, cB = st.columns(2)
                with cA:
                    st.markdown("**Original**")
                    st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
                    st.markdown("**Segmentation Overlay**")
                    st.image(cv2.cvtColor(overlay_mask(bgr, seg_res.glove_mask), cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
                with cB:
                    st.markdown("**Defects Overlay**")
                    over = overlay_mask(bgr, res.glove_mask)
                    draw_list = [
                        d
                        for d in res.defects
                        if d.bbox is not None
                        and passes_threshold(str(d.label), float(d.score), _defect_eval_glove_type(d, eval_glove_type))
                    ]
                    draw_list.sort(key=lambda d: float(d.score), reverse=True)
                    draw_list = draw_list[: int(max_boxes)]
                    over = draw_defects(over, draw_list)
                    st.image(cv2.cvtColor(over, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
                    st.caption(f"seg_method={seg_dbg.chosen_method}")

                if seg_auto_dbg:
                    st.markdown("### Auto Profile Selection")
                    mode = str(seg_auto_dbg.get("mode", ""))
                    if mode == "auto_type_conditional":
                        st.caption(
                            f"profile={seg_auto_dbg.get('chosen_profile', 'balanced')} | "
                            f"top_probe={seg_auto_dbg.get('top_probe_label', 'unknown')} "
                            f"({float(seg_auto_dbg.get('top_probe_score', 0.0)):.2f})"
                        )
                    elif mode == "manual_cfg":
                        st.caption("profile=manual sliders")
                    trials = seg_auto_dbg.get("tried_profiles")
                    if isinstance(trials, list) and trials:
                        st.dataframe(pd.DataFrame(trials), use_container_width=True, hide_index=True)

                st.markdown("### Segmentation Debug")
                if seg_dbg.scored:
                    st.dataframe(pd.DataFrame(seg_dbg.scored).head(10), use_container_width=True, hide_index=True)
                if seg_dbg.candidate_validity:
                    rows_valid = [{"method": k, **v} for k, v in seg_dbg.candidate_validity.items()]
                    st.markdown("**Candidate validity**")
                    st.dataframe(pd.DataFrame(rows_valid), use_container_width=True, hide_index=True)
                st.markdown("**Refine diagnostics**")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"kind": "carve", **seg_dbg.carve_stats},
                            {"kind": "prune", **seg_dbg.prune_stats},
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                c1, c2, c3 = st.columns(3)
                with c1:
                    heat = cv2.applyColorMap(seg_dbg.d_bg_u8, cv2.COLORMAP_TURBO)
                    st.markdown("**d_bg heatmap**")
                    st.image(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
                with c2:
                    st.markdown("**Edges**")
                    st.image(cv2.cvtColor(seg_dbg.edges_u8, cv2.COLOR_GRAY2RGB), channels="RGB", width="stretch")
                with c3:
                    st.markdown("**Gradient**")
                    st.image(cv2.cvtColor(seg_dbg.grad_u8, cv2.COLOR_GRAY2RGB), channels="RGB", width="stretch")


if __name__ == "__main__":
    main()
