from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from gdd.core.anomaly import AnomalyMaps
from gdd.core.defect_detectors import detect_defects
from gdd.core.pipeline import GDDPipeline
from gdd.core.segmentation import SegmentationConfig, SegmentationResult
from gdd.core.types import Defect


def _dummy_image_and_masks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bgr = np.zeros((48, 48, 3), dtype=np.uint8)
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[8:40, 10:38] = 255
    return bgr, mask, mask.copy()


def _dummy_anomaly(mask: np.ndarray) -> AnomalyMaps:
    z = np.zeros(mask.shape, dtype=np.float32)
    return AnomalyMaps(color=z, texture=z, edges=z, combined=z)


class AllModeConsistencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bgr, self.mask, self.mask_filled = _dummy_image_and_masks()
        self.cuff_defects = [
            Defect(label="improper_roll", score=0.83, bbox=None),
            Defect(label="incomplete_beading", score=0.79, bbox=None),
        ]
        self.quality = {
            "area_frac": 0.35,
            "extent": 0.55,
            "border_touch": 0.01,
            "edge_align": 0.01,
            "ok_surface": 0.0,
            "ok_finger": 0.0,
        }

    def _detect_with_stubbed_groups(self, glove_type: str, allowed_labels: set[str] | None) -> tuple[list[Defect], int]:
        with (
            patch("gdd.core.defect_detectors.build_anomaly_maps", return_value=_dummy_anomaly(self.mask)),
            patch("gdd.core.defect_detectors._mask_quality", return_value=self.quality),
            patch("gdd.core.defect_detectors._detect_holes", return_value=[]),
            patch("gdd.core.defect_detectors._edge_fold_wrinkle", return_value=[]),
            patch("gdd.core.defect_detectors._inside_out", return_value=[]),
            patch("gdd.core.defect_detectors._finger_count_anomaly", return_value=[]),
            patch("gdd.core.defect_detectors._spot_stain_discoloration", return_value=[]),
            patch("gdd.core.defect_detectors._discoloration_only", return_value=[]),
            patch("gdd.core.defect_detectors._roll_and_beading", return_value=list(self.cuff_defects)) as cuff_mock,
        ):
            defects, _ = detect_defects(
                self.bgr,
                self.mask,
                self.mask_filled,
                glove_type=glove_type,
                allowed_labels=allowed_labels,
            )
        return defects, cuff_mock.call_count

    def test_all_mode_runs_cuff_detection_for_all_supported_glove_types(self) -> None:
        for glove_type in ("latex", "fabric", "leather"):
            with self.subTest(glove_type=glove_type):
                defects, cuff_calls = self._detect_with_stubbed_groups(glove_type, allowed_labels=None)
                self.assertEqual(cuff_calls, 1)
                self.assertEqual({d.label for d in defects}, {"improper_roll", "incomplete_beading"})

    def test_all_mode_and_explicit_cuff_selection_share_same_detector_path(self) -> None:
        for glove_type in ("fabric", "leather"):
            with self.subTest(glove_type=glove_type):
                all_defects, all_calls = self._detect_with_stubbed_groups(glove_type, allowed_labels=None)
                roll_only, roll_calls = self._detect_with_stubbed_groups(glove_type, allowed_labels={"improper_roll"})
                bead_only, bead_calls = self._detect_with_stubbed_groups(glove_type, allowed_labels={"incomplete_beading"})

                self.assertEqual(all_calls, 1)
                self.assertEqual(roll_calls, 1)
                self.assertEqual(bead_calls, 1)
                self.assertEqual({d.label for d in all_defects}, {"improper_roll", "incomplete_beading"})
                self.assertEqual([d.label for d in roll_only], ["improper_roll"])
                self.assertEqual([d.label for d in bead_only], ["incomplete_beading"])

    def test_pipeline_infer_keeps_single_active_glove_type_and_includes_cuff_defects_in_all_mode(self) -> None:
        seg_res = SegmentationResult(glove_mask=self.mask, glove_mask_filled=self.mask_filled, method="mock")

        with (
            patch("gdd.core.pipeline.segment_glove", return_value=seg_res),
            patch("gdd.core.defect_detectors.build_anomaly_maps", return_value=_dummy_anomaly(self.mask)),
            patch("gdd.core.defect_detectors._mask_quality", return_value=self.quality),
            patch("gdd.core.defect_detectors._detect_holes", return_value=[]),
            patch("gdd.core.defect_detectors._edge_fold_wrinkle", return_value=[]),
            patch("gdd.core.defect_detectors._inside_out", return_value=[]),
            patch("gdd.core.defect_detectors._finger_count_anomaly", return_value=[]),
            patch("gdd.core.defect_detectors._spot_stain_discoloration", return_value=[]),
            patch("gdd.core.defect_detectors._discoloration_only", return_value=[]),
            patch("gdd.core.defect_detectors._roll_and_beading", return_value=list(self.cuff_defects)),
        ):
            pipeline = GDDPipeline(glove_type_model=None)
            for forced_type in ("latex", "fabric", "leather"):
                with self.subTest(forced_type=forced_type):
                    res = pipeline.infer(
                        self.bgr,
                        allowed_labels=None,
                        seg_cfg=SegmentationConfig(),
                        force_glove_type=forced_type,
                    )
                    self.assertEqual(res.glove_type, "unknown")
                    self.assertEqual({d.label for d in res.defects}, {"improper_roll", "incomplete_beading"})

    def test_load_default_tracks_glove_type_model_path_and_status(self) -> None:
        pipeline = GDDPipeline.load_default()
        self.assertEqual(pipeline.glove_type_model_path.name, "glove_type.joblib")
        if pipeline.glove_type_model_path.exists():
            self.assertIsNotNone(pipeline.glove_type_model)
            self.assertIsNone(pipeline.glove_type_model_error)
        else:
            self.assertIsNone(pipeline.glove_type_model)
            self.assertIsNotNone(pipeline.glove_type_model_error)
            self.assertIn("glove_type.joblib", str(pipeline.glove_type_model_error))


if __name__ == "__main__":
    unittest.main()
