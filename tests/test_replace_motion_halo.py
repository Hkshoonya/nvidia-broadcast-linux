"""Replace regressions using synthetic frames and a deterministic alpha backend."""

import unittest
from unittest import mock

import cv2
import numpy as np

from nvbroadcast.video.effects import VideoEffects


class _SyntheticAlphaBackend:
    _lowres_refined = False

    def __init__(self, alpha):
        self.alpha = alpha

    def infer(self, _frame, _width, _height):
        return self.alpha.copy()


class _NumpyFusedCupy:
    """Exercise fused argument plumbing without a CUDA device or camera."""

    # Host mattes must not be mistaken for a device-resident matte.
    class ndarray:
        pass

    float32 = np.float32
    int32 = np.int32
    uint8 = np.uint8
    asarray = staticmethod(np.asarray)
    asnumpy = staticmethod(np.asarray)
    empty_like = staticmethod(np.empty_like)
    zeros = staticmethod(np.zeros)
    ones = staticmethod(np.ones)

    class cuda:
        class Stream:
            class null:
                @staticmethod
                def synchronize():
                    pass


class ReplaceMotionHaloTests(unittest.TestCase):
    @staticmethod
    def _scene():
        """Dark subject, translucent bright-background edge, and a white shirt."""
        height, width = 360, 640
        alpha = np.zeros((height, width), dtype=np.float32)
        alpha[72:288, 240:400] = 1.0
        alpha[72:288, 236:240] = 0.75

        subject = np.full((height, width, 3), 24, dtype=np.float32)
        subject[216:270, 280:360] = 240
        frame = np.empty((height, width, 4), dtype=np.uint8)
        frame[:, :, :3] = (
            subject * alpha[:, :, np.newaxis]
            + 255 * (1.0 - alpha[:, :, np.newaxis])
        ).astype(np.uint8)
        frame[:, :, 3] = 255
        return frame, alpha

    @staticmethod
    def _effects(alpha):
        effects = VideoEffects(compositing="cpu")
        effects._quality = "ultra"
        effects._bg_mode = "replace"
        # Match DocZeus temporal settings without loading CuPy or using a GPU.
        effects._use_fused_kernel = True
        effects._bg_removal_enabled = True
        effects._initialized = True
        effects._backend = _SyntheticAlphaBackend(alpha)
        effects._learned_refiners = {}
        effects._bg_image = np.zeros((*alpha.shape, 4), dtype=np.uint8)
        effects._bg_image[:, :, 3] = 255
        effects._refresh_temporal_strength()
        effects.update_edge_params(
            dilate_size=3,
            blur_size=5,
            sigmoid_strength=14,
            sigmoid_midpoint=0.45,
        )
        return effects

    def test_moving_replace_edge_cleans_source_color_without_eroding_subject(self):
        frame, alpha = self._scene()
        effects = self._effects(alpha)
        height, width = alpha.shape

        for shift in (0, 1, 2, 8):
            with self.subTest(translation=shift):
                current_frame = np.roll(frame, shift, axis=1)
                effects._backend.alpha = np.roll(alpha, shift, axis=1)

                output = effects.process_frame_array(current_frame, width, height)
                final_matte = effects.latest_final_matte_u8(width, height)
                edge_x = 237 + shift

                self.assertGreaterEqual(
                    int(final_matte[180, edge_x]), 250,
                    "Color cleanup must not hide the fringe by eroding the subject.",
                )
                self.assertLessEqual(
                    int(output[180, edge_x, 0]), 60,
                    "Sharpening the matte must not retain the bright camera mixture "
                    "as opaque foreground (the uncorrected edge is 81).",
                )
                self.assertTrue(
                    np.all(output[230:255, 300 + shift:340 + shift, :3] == 240),
                    "True opaque white clothing must retain its original color.",
                )
                if shift == 8:
                    self.assertEqual(int(final_matte[180, 237]), 0)
                    self.assertTrue(np.all(output[180, 237, :3] == 0))

    def test_dark_edge_next_to_white_clothing_does_not_brighten_when_translated(self):
        results = []
        for shift in range(4):
            with self.subTest(translation=shift):
                edge_x = 16 + shift
                frame = np.full((64, 64, 4), 24, dtype=np.uint8)
                frame[:, :, 3] = 255
                alpha = np.zeros((64, 64), dtype=np.float32)
                alpha[:, edge_x] = 0.5
                alpha[:, edge_x + 1:edge_x + 32] = 1.0
                # The one-pixel opaque dark border belongs to the same surface
                # as the translucent edge. White clothing starts beyond it.
                frame[:, edge_x + 2:edge_x + 32, :3] = 240
                effects = self._effects(alpha)

                cleaned = effects._prepare_replace_foreground(frame, alpha)
                results.append(int(cleaned[32, edge_x, 0]))

                self.assertTrue(
                    np.all(cleaned[32, edge_x, :3] == 24),
                    "Uncontaminated dark detail must not borrow white from clothing.",
                )
                self.assertTrue(
                    np.all(cleaned[:, edge_x + 2:edge_x + 32, :3] == 240)
                )

        self.assertEqual(max(results) - min(results), 0)

    def test_large_reference_preserves_detail_and_donor_radius_across_tile_seams(self):
        for edge_x in (127, 128, 129, 511, 512, 513):
            with self.subTest(edge_x=edge_x):
                # This exceeds the tiled-path threshold; partial last tiles
                # also exercise padding for dimensions not divisible by 128.
                frame = np.full((530, 550, 4), 255, dtype=np.uint8)
                alpha = np.zeros(frame.shape[:2], dtype=np.float32)
                alpha[16:145, edge_x] = 0.5
                alpha[16:145, edge_x + 1:edge_x + 34] = 1.0
                frame[16:145, edge_x + 1, :3] = 24
                frame[16:145, edge_x + 2:edge_x + 34, :3] = 240
                # Both a mixed bright edge and a legitimate dark edge share
                # the same narrow opaque dark border beside white clothing.
                frame[16:80, edge_x, :3] = 81
                frame[80:145, edge_x, :3] = 24

                # A donor ten pixels away is usable across a tile boundary;
                # the adjacent pixel eleven pixels away must retain its RGB.
                alpha[300, edge_x - 1:edge_x + 1] = 0.5
                frame[300, edge_x - 1:edge_x + 1, :3] = 110
                alpha[300, edge_x + 10] = 1.0
                frame[300, edge_x + 10, :3] = 42
                # No close donor exists for this separate translucent patch.
                alpha[400:435, 400:435] = 0.5
                frame[400:435, 400:435, :3] = 110

                reference = VideoEffects._nearest_foreground_color(frame, alpha, 0.94)

                self.assertTrue(np.all(reference[16:145, edge_x, :3] == 24))
                self.assertTrue(np.all(reference[16:145, edge_x + 1, :3] == 24))
                self.assertTrue(
                    np.all(reference[16:145, edge_x + 2:edge_x + 34, :3] == 240)
                )
                self.assertTrue(np.all(reference[300, edge_x, :3] == 42))
                self.assertTrue(np.all(reference[300, edge_x - 1, :3] == 110))
                np.testing.assert_array_equal(
                    reference[400:435, 400:435], frame[400:435, 400:435]
                )
                self.assertTrue(np.all(reference[:, :, 3] == 255))

    def test_cleanup_preserves_source_color_when_no_opaque_donor_exists(self):
        frame = np.full((48, 48, 4), 110, dtype=np.uint8)
        frame[:, :, 3] = 255
        alpha = np.zeros((48, 48), dtype=np.float32)
        alpha[12:36, 12:36] = 0.5
        effects = self._effects(alpha)

        cleaned = effects._prepare_replace_foreground(frame, alpha)

        self.assertTrue(
            np.array_equal(cleaned, frame),
            "Missing foreground evidence must not manufacture a black donor.",
        )

    def test_cpu_and_fused_paths_clean_source_color_but_blend_final_opacity(self):
        frame, alpha = self._scene()
        cpu = self._effects(alpha)
        fused = self._effects(alpha)
        fused._cupy = _NumpyFusedCupy()
        height, width = alpha.shape
        kernel_calls = []

        def composite_kernel(_grid, _block, args):
            foreground, background, blend_alpha = args[:3]
            output = args[6]
            kernel_calls.append((foreground.copy(), blend_alpha.copy(), int(args[12])))
            blend = blend_alpha[:, :, np.newaxis]
            output[:] = np.clip(
                foreground.astype(np.float32) * blend
                + background.astype(np.float32) * (1.0 - blend),
                0, 255,
            ).astype(np.uint8)
            output[:, :, 3] = 255

        with mock.patch(
            "nvbroadcast.video.effects._get_fused_kernel", return_value=composite_kernel
        ):
            for shift in (0, 1):
                with self.subTest(translation=shift):
                    current_frame = np.roll(frame, shift, axis=1)
                    current_alpha = np.roll(alpha, shift, axis=1)
                    cpu._backend.alpha = current_alpha
                    fused._backend.alpha = current_alpha
                    expected = cpu.process_frame_array(current_frame, width, height)
                    actual = fused.process_frame_array(current_frame, width, height)
                    edge_x = 237 + shift

                    self.assertEqual(len(kernel_calls), shift + 1)
                    uploaded, blend_alpha, gpu_despill = kernel_calls[-1]
                    self.assertLessEqual(int(uploaded[180, edge_x, 0]), 60)
                    self.assertGreater(float(blend_alpha[180, edge_x]), 0.99)
                    self.assertEqual(gpu_despill, 0)
                    np.testing.assert_allclose(actual, expected, atol=1, rtol=0)
                    self.assertTrue(np.all(actual[230:255, 300:340, :3] == 240))

    def test_cached_source_and_refined_mattes_resize_together(self):
        frame, alpha = self._scene()
        effects = self._effects(alpha)
        height, width = alpha.shape
        effects.process_frame_array(frame, width, height)
        enlarged = cv2.resize(frame, (width * 2, height * 2), interpolation=cv2.INTER_NEAREST)

        with mock.patch.object(
            effects, "_prepare_replace_foreground", wraps=effects._prepare_replace_foreground
        ) as prepare:
            output = effects.composite_only_array(enlarged, width * 2, height * 2)

        prepare.assert_called_once()
        cleanup_alpha = prepare.call_args.args[1]
        expected_source = cv2.resize(
            alpha, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR
        )
        np.testing.assert_array_equal(cleanup_alpha, expected_source)
        final_matte = effects.latest_final_matte_u8(width * 2, height * 2)
        self.assertGreaterEqual(int(final_matte[360, 475]), 250)
        self.assertLessEqual(int(output[360, 475, 0]), 60)
        self.assertTrue(np.all(output[460:510, 600:680, :3] == 240))

    def test_in_place_refinement_cannot_overwrite_source_color_evidence(self):
        frame, alpha = self._scene()
        effects = self._effects(alpha)
        height, width = alpha.shape
        refined_inputs = []

        def refine_in_place(value):
            refined_inputs.append(value)
            value[value > 0.5] = 1.0
            return value

        with mock.patch.object(effects, "_refine_alpha", side_effect=refine_in_place), \
                mock.patch.object(
                    effects, "_prepare_replace_foreground",
                    wraps=effects._prepare_replace_foreground,
                ) as prepare:
            output = effects.process_frame_array(frame, width, height)

        prepare.assert_called_once()
        cleanup_alpha = prepare.call_args.args[1]
        np.testing.assert_array_equal(cleanup_alpha, alpha)
        self.assertFalse(np.shares_memory(cleanup_alpha, refined_inputs[0]))
        self.assertEqual(float(refined_inputs[0][180, 237]), 1.0)
        self.assertGreaterEqual(int(effects.latest_final_matte_u8(width, height)[180, 237]), 250)
        self.assertLessEqual(int(output[180, 237, 0]), 60)

    def test_backend_handoff_retains_cleanup_for_its_retained_matte(self):
        frame, alpha = self._scene()
        effects = self._effects(alpha)
        height, width = alpha.shape
        before = effects.process_frame_array(frame, width, height)

        effects._prepare_backend_handoff()
        after = effects.composite_only_array(frame, width, height)

        self.assertLessEqual(int(before[180, 237, 0]), 60)
        self.assertLessEqual(
            int(after[180, 237, 0]), 60,
            "Keeping the displayed matte during handoff must also keep its "
            "matching foreground evidence.",
        )

    def test_unpaired_commit_does_not_reuse_an_older_source_matte(self):
        frame, alpha = self._scene()
        effects = self._effects(alpha)
        height, width = alpha.shape
        effects.process_frame_array(frame, width, height)

        solid_alpha = alpha.copy()
        solid_alpha[72:288, 236:240] = 1.0
        white_frame = frame.copy()
        white_frame[72:288, 236:240, :3] = 240
        self.assertTrue(effects._commit_alpha(solid_alpha, effects._matte_version))

        output = effects.composite_only_array(white_frame, width, height)

        self.assertTrue(
            np.all(output[180, 237, :3] == 240),
            "A committed solid white detail must not inherit an earlier "
            "translucent source matte.",
        )

    def test_intervening_commit_preserves_the_snapshotted_source_matte(self):
        for entry_point in ("composite_only_array", "process_frame_array"):
            with self.subTest(entry_point=entry_point):
                frame, alpha = self._scene()
                effects = self._effects(alpha)
                height, width = alpha.shape
                effects.process_frame_array(frame, width, height)
                # Exercise the cached path through both public entry points.
                effects._skip_interval = 100
                original_snapshot = effects._matte_color_snapshot
                captured = []

                def snapshot_then_commit():
                    snapshot = original_snapshot()
                    captured.append(snapshot)
                    # A newer inference makes this edge solid. The in-flight
                    # composite still needs the old matte's translucent color
                    # evidence, even though neither cached array is current.
                    newer_source = alpha.copy()
                    newer_source[72:288, 236:240] = 1.0
                    effects._backend.alpha = newer_source
                    newer_alpha = effects._run_inference(
                        frame, width, height, snapshot[2]
                    )
                    self.assertTrue(effects._commit_alpha(newer_alpha, snapshot[2]))
                    self.assertIsNot(newer_alpha, snapshot[0])
                    return snapshot

                with mock.patch.object(
                    effects, "_matte_color_snapshot", side_effect=snapshot_then_commit
                ), mock.patch.object(
                    effects, "_prepare_replace_foreground",
                    wraps=effects._prepare_replace_foreground,
                ) as prepare:
                    output = getattr(effects, entry_point)(frame, width, height)

                self.assertEqual(len(captured), 1)
                prepare.assert_called_once()
                np.testing.assert_array_equal(prepare.call_args.args[1], alpha)
                self.assertLessEqual(
                    int(output[180, 237, 0]), 60,
                    "An intervening inference commit must not restore the "
                    "bright camera mixture by dropping the source matte.",
                )

    def test_reset_rejects_late_inference_without_leaking_its_source_matte(self):
        frame, alpha = self._scene()
        effects = self._effects(alpha)
        height, width = alpha.shape
        old_version = effects._matte_version
        late_alpha = effects._run_inference(frame, width, height, old_version)
        self.assertIsNotNone(late_alpha)

        effects.reset_cached_mattes()
        self.assertFalse(effects._commit_alpha(late_alpha, old_version))
        solid_alpha = alpha.copy()
        solid_alpha[72:288, 236:240] = 1.0
        white_frame = frame.copy()
        white_frame[72:288, 236:240, :3] = 240
        self.assertTrue(effects._commit_alpha(solid_alpha, effects._matte_version))

        output = effects.composite_only_array(white_frame, width, height)

        self.assertTrue(
            np.all(output[180, 237, :3] == 240),
            "Rejected inference must not attach its old source matte to the "
            "newly committed frame.",
        )


if __name__ == "__main__":
    unittest.main()
