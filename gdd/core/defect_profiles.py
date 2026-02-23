from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DefectTypeProfile:
    name: str
    anomaly_color_weight: float
    anomaly_texture_weight: float
    anomaly_edge_weight: float
    edge_blur_ksize: int
    enable_specular_suppression: bool
    blob_threshold_mode: str  # {"otsu","percentile"}
    blob_threshold_percentile: float
    blob_threshold_offset: float
    min_spot_count: int
    plastic_min_edge_strength: float
    min_mask_extent_for_surface: float


_PROFILE_BALANCED = DefectTypeProfile(
    name="balanced",
    anomaly_color_weight=0.55,
    anomaly_texture_weight=0.25,
    anomaly_edge_weight=0.20,
    edge_blur_ksize=5,
    enable_specular_suppression=False,
    blob_threshold_mode="otsu",
    blob_threshold_percentile=94.0,
    blob_threshold_offset=10.0,
    min_spot_count=8,
    plastic_min_edge_strength=0.08,
    min_mask_extent_for_surface=0.22,
)

_PROFILE_LATEX = DefectTypeProfile(
    name="latex",
    anomaly_color_weight=0.60,
    anomaly_texture_weight=0.10,
    anomaly_edge_weight=0.30,
    edge_blur_ksize=5,
    enable_specular_suppression=True,
    blob_threshold_mode="otsu",
    blob_threshold_percentile=93.0,
    blob_threshold_offset=8.0,
    min_spot_count=8,
    plastic_min_edge_strength=0.07,
    min_mask_extent_for_surface=0.24,
)

_PROFILE_LEATHER = DefectTypeProfile(
    name="leather",
    anomaly_color_weight=0.65,
    anomaly_texture_weight=0.20,
    anomaly_edge_weight=0.15,
    edge_blur_ksize=9,
    enable_specular_suppression=True,
    blob_threshold_mode="percentile",
    blob_threshold_percentile=95.0,
    blob_threshold_offset=0.0,
    min_spot_count=10,
    plastic_min_edge_strength=0.11,
    min_mask_extent_for_surface=0.25,
)

_PROFILE_FABRIC = DefectTypeProfile(
    name="fabric",
    anomaly_color_weight=0.70,
    anomaly_texture_weight=0.05,
    anomaly_edge_weight=0.25,
    edge_blur_ksize=11,
    enable_specular_suppression=False,
    blob_threshold_mode="percentile",
    blob_threshold_percentile=97.0,
    blob_threshold_offset=0.0,
    min_spot_count=12,
    plastic_min_edge_strength=0.12,
    min_mask_extent_for_surface=0.26,
)


def get_defect_profile(glove_type: str | None) -> DefectTypeProfile:
    g = str(glove_type or "").strip().lower()
    if g == "latex":
        return _PROFILE_LATEX
    if g == "leather":
        return _PROFILE_LEATHER
    if g == "fabric":
        return _PROFILE_FABRIC
    return _PROFILE_BALANCED
