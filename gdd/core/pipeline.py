from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .defect_detectors import detect_defects
from .features import glove_type_features
from .glove_type_model import GloveTypeModel, load_glove_type_model
from .preprocess import preprocess
from .segmentation import SegmentationConfig, SegmentationResult, segment_glove
from .types import InferenceResult


DEFAULT_GLOVE_TYPE_MODEL_PATH = Path("gdd/models/glove_type.joblib")
DEFAULT_SEG_TOPK = 2
DEFAULT_SEG_CONFIDENCE_THR = 0.75
KNOWN_GLOVE_TYPES = ("latex", "leather", "fabric")


def _probe_seg_cfg() -> SegmentationConfig:
    return SegmentationConfig(
        profile_name="balanced",
        gap_carve_mode="hybrid",
        edge_recover_enabled=False,
        upright_enabled=False,
        valley_carve_enabled=False,
        webbing_restore_enabled=False,
        webbing_convexity_enabled=False,
        reach_carve_enabled=False,
        grabcut_open_k=3,
        grabcut_close_k=5,
    )


def _profile_seg_cfg(glove_type: str) -> SegmentationConfig:
    g = str(glove_type).strip().lower()
    if g == "leather":
        return SegmentationConfig(
            profile_name="leather",
            gap_carve_mode="hybrid",
            chroma_min_area_frac=0.10,
            chroma_min_iou_silhouette=0.12,
            shadow_strictness=0.85,
            reach_carve_enabled=True,
            reach_carve_strength=1.30,
            webbing_trigger_ratio=0.10,
            gap_geometry_fallback_solidity=0.88,
            bg_like_percentile=95.0,
            bg_like_margin=4.0,
            global_halo_peel_trigger_ratio=0.18,
            global_halo_peel_max_remove_frac=0.08,
            tip_restore_enabled=True,
            tip_restore_max_add_frac=0.040,
            grabcut_open_k=1,
            grabcut_erode_frac=0.005,   # tiny erosion: keep fingertip pixels as sure_fg
            grabcut_dilate_frac=0.045,  # large dilation: cover rounded tips as prob_fg
            edge_recover_enabled=True,
            edge_recover_kd_scale=1.20,
            edge_recover_ring_scale=1.15,
            edge_recover_max_area_mul=1.28,
            # ── Leather type-specific strategy ──
            leather_saturation_candidate=True,
            candidate_boost=(("chroma_fg", 0.10), ("bg_dist", 0.06), ("saturation_fg", 0.12)),
            candidate_suppress=(("texture_fg", 0.04),),
            edge_finger_separation_enabled=True,
        )
    if g == "fabric":
        return SegmentationConfig(
            profile_name="fabric",
            gap_carve_mode="hybrid",
            chroma_min_area_frac=0.20,
            chroma_min_iou_silhouette=0.25,
            shadow_strictness=1.15,
            reach_carve_enabled=True,
            reach_carve_strength=1.20,  # boosted: aggressively carve inter-finger gaps
            webbing_trigger_ratio=0.08,  # lower trigger: more aggressive webbing carve
            webbing_restore_enabled=True,
            bg_like_percentile=97.0,
            bg_like_margin=7.0,
            global_halo_peel_trigger_ratio=0.11,
            global_halo_peel_max_remove_frac=0.14,
            tip_restore_enabled=True,  # enable for all types now
            tip_restore_max_add_frac=0.020,
            edge_recover_enabled=True,
            edge_recover_kd_scale=0.90,
            edge_recover_ring_scale=0.90,
            edge_recover_max_area_mul=1.18,
            grabcut_open_k=3,
            grabcut_close_k=3,
            grabcut_erode_frac=0.008,
            grabcut_dilate_frac=0.015,
            grabcut_candidate_cap_frac=0.025,  # cap GrabCut to dilated candidate, preventing gap fill
            chan_vese_enabled=True,
            # ── Fabric type-specific strategy ──
            fabric_variance_candidate=True,
            candidate_boost=(("texture_fg", 0.10), ("variance_fg", 0.12)),
            candidate_suppress=(("chroma_fg", 0.05),),
            edge_finger_separation_enabled=True,
        )
    if g == "latex":
        return SegmentationConfig(
            profile_name="latex",
            gap_carve_mode="hybrid",
            chroma_min_area_frac=0.20,
            chroma_min_iou_silhouette=0.25,
            shadow_strictness=0.95,
            reach_carve_enabled=True,
            reach_carve_strength=0.70,
            webbing_trigger_ratio=0.10,
            bg_like_percentile=98.0,  # raised: avoid peeling near-white glove pixels
            bg_like_margin=10.0,  # raised: higher margin = less aggressive bg classification
            global_halo_peel_trigger_ratio=0.20,  # raised from 0.12: was over-peeling latex
            global_halo_peel_max_remove_frac=0.06,  # reduced: limit how much can be peeled
            global_halo_peel_dt_frac=0.020,  # thinner peel band
            tip_restore_enabled=True,  # ENABLED: was False, causing clipped tips
            tip_restore_max_add_frac=0.035,
            grabcut_open_k=1,
            grabcut_erode_frac=0.004,   # minimal erosion: near-white tips are thin
            grabcut_dilate_frac=0.050,  # very large dilation: cover translucent tip boundaries
            grabcut_trimap_enabled=True,
            grabcut_trimap_only_if_low_confidence=False,
            chan_vese_enabled=True,
            edge_recover_enabled=True,
            edge_recover_kd_scale=1.50,
            edge_recover_ring_scale=1.40,
            edge_recover_max_area_mul=1.35,
            # ── Latex type-specific strategy ──
            latex_edge_multiscale=True,
            candidate_boost=(("edge_closed", 0.08), ("edge_closed_strong", 0.08), ("latex_ms_edge", 0.14)),
            candidate_suppress=(("chroma_fg", 0.08),),
            edge_finger_separation_enabled=True,
        )
    return SegmentationConfig(profile_name="balanced", gap_carve_mode="hybrid")


class GDDPipeline:
    def __init__(self, glove_type_model: GloveTypeModel | None = None):
        self.glove_type_model = glove_type_model
        self.last_seg_debug_info: dict[str, Any] | None = None

    @classmethod
    def load_default(cls) -> "GDDPipeline":
        model = None
        if DEFAULT_GLOVE_TYPE_MODEL_PATH.exists():
            try:
                model = load_glove_type_model(DEFAULT_GLOVE_TYPE_MODEL_PATH)
            except Exception:
                # Model may be stale (wrong classes or wrong feature length). Treat as missing.
                model = None
        return cls(glove_type_model=model)

    @staticmethod
    def _mask_quality_score(bgr: np.ndarray, mask_u8: np.ndarray) -> tuple[float, dict[str, float]]:
        mask01 = (mask_u8 > 0).astype(np.uint8)
        h, w = mask01.shape[:2]
        area = float(mask01.sum())
        if area <= 0:
            return -1e9, {"reason_empty": 1.0}

        area_frac = area / float(h * w + 1e-6)
        if area_frac < 0.03 or area_frac > 0.90:
            return -1e9, {"area_frac": float(area_frac)}

        x, y, ww, hh = cv2.boundingRect(mask01.astype(np.uint8))
        extent = float(area) / float(ww * hh + 1e-6) if ww > 0 and hh > 0 else 0.0

        band = max(5, int(round(min(h, w) * 0.04)))
        border = np.zeros_like(mask01, dtype=np.uint8)
        border[:band, :] = 1
        border[-band:, :] = 1
        border[:, :band] = 1
        border[:, -band:] = 1
        border_touch = float((mask01 & border).sum()) / float(area + 1e-6)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        k = 5
        dil = cv2.dilate(mask01, np.ones((k, k), np.uint8), iterations=1)
        ero = cv2.erode(mask01, np.ones((k, k), np.uint8), iterations=1)
        ring = (dil > 0).astype(np.uint8) - (ero > 0).astype(np.uint8)
        edge_align = float(((ring > 0) & (edges > 0)).sum()) / float(ring.sum() + 1e-6)

        score = 0.0
        score += 1.4 * float(edge_align)
        score += 1.4 * float(extent)
        score -= 2.8 * float(max(0.0, border_touch - 0.05))
        score -= 1.8 * float(max(0.0, 0.08 - extent))
        score -= 1.8 * float(max(0.0, 0.05 - area_frac))
        score -= 1.8 * float(max(0.0, area_frac - 0.80))

        metrics = {
            "area_frac": float(area_frac),
            "extent": float(extent),
            "border_touch": float(border_touch),
            "edge_align": float(edge_align),
        }
        return float(score), metrics

    @staticmethod
    def _material_cue_scores(bgr: np.ndarray, mask_u8: np.ndarray) -> dict[str, float]:
        mask01 = (mask_u8 > 0).astype(np.uint8)
        if int(mask01.sum()) < 400:
            return {"fabric": 0.0, "leather": 0.0, "interior_edge_density": 0.0, "laplace_std": 0.0, "sat_mean": 0.0}

        interior = cv2.erode(mask01, np.ones((9, 9), np.uint8), iterations=1)
        if int(interior.sum()) < 200:
            interior = mask01

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        edge_density = float(((edges > 0) & (interior > 0)).sum()) / float(interior.sum() + 1e-6)

        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        lap_vals = np.abs(lap[interior > 0]).astype(np.float32)
        lap_std = float(lap_vals.std()) if lap_vals.size else 0.0

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat_vals = hsv[:, :, 1][interior > 0].astype(np.float32)
        sat_mean = float(sat_vals.mean()) if sat_vals.size else 0.0

        fabric_score = 0.60 * min(1.0, edge_density / 0.12) + 0.40 * min(1.0, lap_std / 42.0)
        leather_score = 0.55 * min(1.0, max(0.0, 0.12 - edge_density) / 0.12) + 0.45 * min(1.0, sat_mean / 48.0)
        return {
            "fabric": float(fabric_score),
            "leather": float(leather_score),
            "interior_edge_density": float(edge_density),
            "laplace_std": float(lap_std),
            "sat_mean": float(sat_mean),
        }

    def get_profile_seg_cfg(self, glove_type: str) -> SegmentationConfig:
        return _profile_seg_cfg(glove_type)

    def _predict_glove_type_probs(self, bgr_p: np.ndarray, mask_u8: np.ndarray, mask_filled_u8: np.ndarray) -> dict[str, float]:
        if self.glove_type_model is None:
            return {}
        feats = glove_type_features(bgr_p, mask_u8, mask_filled_u8)
        try:
            probs = self.glove_type_model.predict_proba(feats)
        except Exception:
            return {}
        out: dict[str, float] = {}
        for gt in KNOWN_GLOVE_TYPES:
            if gt in probs:
                out[gt] = float(probs[gt])
        return out

    def _resolve_glove_type(
        self,
        trial_probs: dict[str, float],
        cue_scores: dict[str, float],
        chosen_profile: str,
    ) -> tuple[str, float, dict[str, float]]:
        blended: dict[str, float] = {gt: float(trial_probs.get(gt, 0.0)) for gt in KNOWN_GLOVE_TYPES}
        profile = str(chosen_profile).strip().lower()
        if profile in blended:
            blended[profile] += 0.18

        blended["fabric"] += 0.42 * float(cue_scores.get("fabric", 0.0))
        blended["fabric"] -= 0.10 * float(cue_scores.get("leather", 0.0))
        blended["leather"] += 0.18 * float(cue_scores.get("leather", 0.0))
        blended["leather"] -= 0.20 * float(cue_scores.get("fabric", 0.0))

        if profile == "fabric" and float(cue_scores.get("fabric", 0.0)) >= 0.75:
            blended["fabric"] += 0.18
        if profile == "leather" and float(cue_scores.get("leather", 0.0)) >= 0.65:
            blended["leather"] += 0.10

        best_label = max(KNOWN_GLOVE_TYPES, key=lambda gt: float(blended.get(gt, 0.0)))
        total = float(sum(max(0.0, float(v)) for v in blended.values()))
        if total <= 1e-6:
            return "unknown", 0.0, blended
        score = float(max(0.0, blended.get(best_label, 0.0)) / total)
        return str(best_label), score, blended

    def _auto_segment(self, bgr_p: np.ndarray) -> tuple[SegmentationResult, dict[str, Any]]:
        debug: dict[str, Any] = {
            "mode": "auto_type_conditional",
            "model_available": bool(self.glove_type_model is not None),
        }
        if self.glove_type_model is None:
            seg = segment_glove(bgr_p, cfg=None)
            debug["reason"] = "model_missing"
            debug["tried_profiles"] = [{"profile": "balanced", "quality": None, "seg_method": str(seg.method)}]
            debug["chosen_profile"] = "balanced"
            return seg, debug

        probe_seg = segment_glove(bgr_p, cfg=_probe_seg_cfg())
        probs = self._predict_glove_type_probs(bgr_p, probe_seg.glove_mask, probe_seg.glove_mask_filled)
        ranked = sorted(probs.items(), key=lambda kv: float(kv[1]), reverse=True)
        debug["probe_type_probs"] = {k: float(v) for k, v in ranked}

        if not ranked:
            try_profiles = list(KNOWN_GLOVE_TYPES)
            top_label = "unknown"
            top_prob = 0.0
        else:
            top_label, top_prob = str(ranked[0][0]), float(ranked[0][1])
            if top_prob >= float(DEFAULT_SEG_CONFIDENCE_THR):
                try_profiles = [top_label]
                # Fabric cuffs and knit textures can be over-confidently pulled toward
                # leather on the probe mask. Always compare against fabric before
                # finalizing a leather-led path.
                if top_label == "leather" and "fabric" not in try_profiles:
                    try_profiles.append("fabric")
            else:
                try_profiles = [gt for gt, _p in ranked[: int(DEFAULT_SEG_TOPK)]]
                if len(try_profiles) < 2:
                    try_profiles = list(KNOWN_GLOVE_TYPES)

        trials: list[dict[str, Any]] = []
        best_seg = None
        best_quality = -1e12
        best_profile = "balanced"
        for profile in try_profiles:
            cfg = _profile_seg_cfg(profile)
            seg = segment_glove(bgr_p, cfg=cfg)
            q, metrics = self._mask_quality_score(bgr_p, seg.glove_mask)
            trial_probs = self._predict_glove_type_probs(bgr_p, seg.glove_mask, seg.glove_mask_filled)
            cue_scores = self._material_cue_scores(bgr_p, seg.glove_mask)
            self_prob = float(trial_probs.get(str(profile), 0.0))
            # Blend segmentation quality with classifier confidence and material cues.
            q += 0.04 * float(self_prob)
            if str(profile) == str(top_label):
                q += 0.04
            if str(profile) == "fabric":
                q += 0.16 * float(cue_scores.get("fabric", 0.0))
                q -= 0.05 * float(cue_scores.get("leather", 0.0))
            elif str(profile) == "leather":
                q += 0.10 * float(cue_scores.get("leather", 0.0))
                q -= 0.08 * float(cue_scores.get("fabric", 0.0))
            row = {
                "profile": str(profile),
                "quality": float(q),
                "seg_method": str(seg.method),
                "trial_self_prob": float(self_prob),
                "trial_type_probs": {k: float(v) for k, v in sorted(trial_probs.items(), key=lambda kv: float(kv[1]), reverse=True)},
                "material_cues": {k: float(v) for k, v in cue_scores.items()},
                **metrics,
            }
            trials.append(row)
            better = q > best_quality
            near_tie = abs(float(q) - float(best_quality)) <= 0.01
            prefer_top = near_tie and str(profile) == str(top_label)
            if better or prefer_top:
                best_quality = float(q)
                best_seg = seg
                best_profile = str(profile)

        if best_seg is None:
            best_seg = segment_glove(bgr_p, cfg=None)
            best_profile = "balanced"

        debug["top_probe_label"] = str(top_label)
        debug["top_probe_score"] = float(top_prob)
        debug["tried_profiles"] = trials
        debug["chosen_profile"] = str(best_profile)
        debug["probe_seg_method"] = str(probe_seg.method)
        return best_seg, debug

    def infer(
        self,
        bgr: np.ndarray,
        focus_only: bool = False,
        allowed_labels: set[str] | None = None,
        seg_cfg: SegmentationConfig | None = None,
        force_glove_type: str | None = None,
    ) -> InferenceResult:
        bgr_p = preprocess(bgr)
        if seg_cfg is not None:
            seg = segment_glove(bgr_p, cfg=seg_cfg)
            self.last_seg_debug_info = {
                "mode": "manual_cfg",
                "chosen_profile": "manual",
                "tried_profiles": [{"profile": "manual", "quality": None, "seg_method": str(seg.method)}],
            }
        else:
            seg, seg_debug = self._auto_segment(bgr_p)
            self.last_seg_debug_info = dict(seg_debug)

        glove_type = "unknown"
        glove_type_score = 0.0
        if self.glove_type_model is not None:
            feats = glove_type_features(bgr_p, seg.glove_mask, seg.glove_mask_filled)
            try:
                trial_probs = self.glove_type_model.predict_proba(feats)
                cue_scores = self._material_cue_scores(bgr_p, seg.glove_mask)
                chosen_profile = "balanced"
                if self.last_seg_debug_info is not None:
                    chosen_profile = str(self.last_seg_debug_info.get("chosen_profile", "balanced"))
                glove_type, glove_type_score, blended_scores = self._resolve_glove_type(trial_probs, cue_scores, chosen_profile)
                if self.last_seg_debug_info is not None:
                    self.last_seg_debug_info["final_type_probs"] = {k: float(v) for k, v in sorted(trial_probs.items(), key=lambda kv: float(kv[1]), reverse=True)}
                    self.last_seg_debug_info["final_material_cues"] = {k: float(v) for k, v in cue_scores.items()}
                    self.last_seg_debug_info["final_blended_type_scores"] = {k: float(v) for k, v in sorted(blended_scores.items(), key=lambda kv: float(kv[1]), reverse=True)}
                    self.last_seg_debug_info["final_glove_type"] = str(glove_type)
                    self.last_seg_debug_info["final_glove_type_score"] = float(glove_type_score)
            except Exception:
                glove_type, glove_type_score = "unknown", 0.0

        detect_glove_type = str(glove_type)
        if force_glove_type is not None and str(force_glove_type).strip():
            detect_glove_type = str(force_glove_type).strip().lower()
            if self.last_seg_debug_info is not None:
                self.last_seg_debug_info["forced_defect_glove_type"] = str(detect_glove_type)
        elif self.last_seg_debug_info is not None:
            self.last_seg_debug_info["defect_glove_type"] = str(detect_glove_type)

        defects, _anom = detect_defects(
            bgr_p,
            seg.glove_mask,
            seg.glove_mask_filled,
            glove_type=detect_glove_type,
            focus_only=bool(focus_only),
            allowed_labels=allowed_labels,
        )

        return InferenceResult(
            glove_mask=seg.glove_mask,
            glove_type=glove_type,
            glove_type_score=float(glove_type_score),
            defects=defects,
        )
