"""Focused tests for training-bundle helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - environment-specific
    np = None

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - environment-specific
    cv2 = None


@unittest.skipIf(np is None or cv2 is None, "numpy/cv2 not installed")
class TrainingBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "onnxruntime" not in sys.modules:
            class _FakeSessionOptions:
                def __init__(self):
                    self.graph_optimization_level = None
                    self.log_severity_level = None

            class _FakeGraphOptimizationLevel:
                ORT_ENABLE_ALL = 0

            class _FakeInferenceSession:
                def __init__(self, *args, **kwargs):
                    pass

            sys.modules["onnxruntime"] = types.SimpleNamespace(
                InferenceSession=_FakeInferenceSession,
                SessionOptions=_FakeSessionOptions,
                GraphOptimizationLevel=_FakeGraphOptimizationLevel,
            )

        script_path = Path("scripts/prepare_matting_training_bundle.py").resolve()
        spec = importlib.util.spec_from_file_location("prepare_matting_training_bundle", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        cls.module = module

    def test_build_trimap_reserves_unknown_band(self):
        alpha = np.zeros((9, 9), dtype=np.float32)
        alpha[1:8, 1:8] = 1.0
        alpha[3:6, 3:6] = 0.85

        trimap = self.module._build_trimap(alpha, fg_threshold=0.97, bg_threshold=0.03, radius=1)

        self.assertEqual(int(trimap[0, 0]), 0, "background should stay background")
        self.assertEqual(int(trimap[4, 4]), 128, "soft interior should remain unknown")
        self.assertEqual(int(trimap[2, 2]), 255, "confident foreground should stay foreground")

    def test_metric_dict_tracks_soft_edge_fraction(self):
        alpha = np.array(
            [
                [0.0, 0.1, 0.9],
                [0.02, 0.5, 0.97],
                [0.0, 0.3, 1.0],
            ],
            dtype=np.float32,
        )

        metrics = self.module._metric_dict(alpha)

        self.assertAlmostEqual(metrics["mid_fraction"], 4 / 9)
        self.assertAlmostEqual(metrics["low_mid_fraction"], 2 / 9)
        self.assertAlmostEqual(metrics["nonzero_fraction"], 7 / 9)


if __name__ == "__main__":
    unittest.main()
