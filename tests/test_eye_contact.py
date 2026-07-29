import unittest

import cv2
import numpy as np

from nvbroadcast.video.eye_contact import (
    EyeContactCorrector,
    _LEFT_EYE,
    _LEFT_IRIS,
    _RIGHT_EYE,
    _RIGHT_IRIS,
)


class _Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class EyeContactTests(unittest.TestCase):
    _WIDTH = 220
    _HEIGHT = 120
    _EYE_CENTERS = ((65, 60), (155, 60))
    _EYE_AXES = (22, 9)

    @classmethod
    def _scene(cls, left_offset=(8, 0), right_offset=(8, 0),
               nose_offset=0, blink_eye=None):
        frame = np.full((cls._HEIGHT, cls._WIDTH, 4), 100, dtype=np.uint8)
        frame[:, :, 3] = 77
        landmarks = [_Point(0.5, 0.5) for _ in range(478)]

        for eye_number, (center, eye_indices, iris_indices, offset) in enumerate((
            (cls._EYE_CENTERS[0], _LEFT_EYE, _LEFT_IRIS, left_offset),
            (cls._EYE_CENTERS[1], _RIGHT_EYE, _RIGHT_IRIS, right_offset),
        )):
            cv2.ellipse(
                frame,
                center,
                cls._EYE_AXES,
                0,
                0,
                360,
                (235, 235, 235, 77),
                -1,
            )
            iris_center = (center[0] + offset[0], center[1] + offset[1])
            cv2.circle(frame, iris_center, 5, (15, 15, 15, 77), -1)

            eye_height = 0 if blink_eye == eye_number else cls._EYE_AXES[1]
            angles = np.linspace(0.0, 2.0 * np.pi, len(eye_indices), endpoint=False)
            for index, angle in zip(eye_indices, angles):
                x = center[0] + cls._EYE_AXES[0] * np.cos(angle)
                y = center[1] + eye_height * np.sin(angle)
                landmarks[index] = _Point(x / cls._WIDTH, y / cls._HEIGHT)

            iris_points = (
                iris_center,
                (iris_center[0] - 4, iris_center[1]),
                (iris_center[0], iris_center[1] - 4),
                (iris_center[0] + 4, iris_center[1]),
                (iris_center[0], iris_center[1] + 4),
            )
            for index, point in zip(iris_indices, iris_points):
                landmarks[index] = _Point(
                    point[0] / cls._WIDTH,
                    point[1] / cls._HEIGHT,
                )

        landmarks[1] = _Point(
            (cls._WIDTH * 0.5 + nose_offset) / cls._WIDTH,
            78 / cls._HEIGHT,
        )
        return frame, landmarks

    @classmethod
    def _iris_centers(cls, frame):
        centers = []
        for center in cls._EYE_CENTERS:
            x1, x2 = center[0] - 18, center[0] + 19
            y1, y2 = center[1] - 7, center[1] + 8
            dark = np.argwhere(frame[y1:y2, x1:x2, 0] < 50)
            if len(dark) == 0:
                raise AssertionError("iris pixels disappeared")
            centers.append(
                np.array((dark[:, 1].mean() + x1, dark[:, 0].mean() + y1))
            )
        return centers

    @classmethod
    def _shifts(cls, corrector, landmarks):
        eyes = [
            corrector._measure_eye(
                landmarks,
                eye_indices,
                iris_indices,
                cls._WIDTH,
                cls._HEIGHT,
            )
            for eye_indices, iris_indices in (
                (_LEFT_EYE, _LEFT_IRIS),
                (_RIGHT_EYE, _RIGHT_IRIS),
            )
        ]
        if any(eye is None for eye in eyes):
            raise AssertionError("test eye geometry was rejected")
        return corrector._binocular_shifts(eyes, landmarks, cls._WIDTH)

    def test_default_mode_is_natural_and_invalid_values_fall_back(self):
        corrector = EyeContactCorrector()

        self.assertEqual(corrector.mode, "natural")
        corrector.mode = "gaze_lock"
        self.assertEqual(corrector.mode, "gaze_lock")
        corrector.mode = "broken"
        self.assertEqual(corrector.mode, "natural")

    def test_default_intensity_moves_shared_gaze_without_inward_convergence(self):
        frame, landmarks = self._scene()
        before = self._iris_centers(frame)
        corrector = EyeContactCorrector()
        corrector.enabled = True
        corrector.intensity = 0.35

        output = corrector.process_frame(frame.copy(), landmarks=landmarks)
        after = self._iris_centers(output)

        self.assertLess(after[0][0], before[0][0] - 3.0)
        self.assertLess(after[1][0], before[1][0] - 3.0)
        self.assertGreater(
            after[1][0] - after[0][0],
            before[1][0] - before[0][0] + 0.5,
        )
        self.assertGreater(
            np.count_nonzero(output[:, :, :3] != frame[:, :, :3]),
            200,
        )
        self.assertTrue(np.array_equal(output[:, :, 3], frame[:, :, 3]))

    def test_shared_target_preserves_natural_binocular_disparity(self):
        frame, landmarks = self._scene(
            left_offset=(-3, 0),
            right_offset=(3, 0),
        )
        before = self._iris_centers(frame)
        corrector = EyeContactCorrector()
        corrector.enabled = True
        corrector.intensity = 1.0

        output = corrector.process_frame(frame.copy(), landmarks=landmarks)
        after = self._iris_centers(output)

        self.assertGreaterEqual(
            after[1][0] - after[0][0],
            before[1][0] - before[0][0],
        )

    def test_zero_intensity_is_a_no_op(self):
        frame, landmarks = self._scene()
        corrector = EyeContactCorrector()
        corrector.enabled = True
        corrector.intensity = 0.0

        output = corrector.process_frame(frame.copy(), landmarks=landmarks)

        self.assertTrue(np.array_equal(output, frame))

    def test_blink_skips_both_eyes_and_resets_smoothing(self):
        frame, landmarks = self._scene()
        corrector = EyeContactCorrector()
        corrector.enabled = True
        corrector.process_frame(frame.copy(), landmarks=landmarks)
        self.assertIsNotNone(corrector._smoothed_correction)

        blink_frame, blink_landmarks = self._scene(blink_eye=0)
        output = corrector.process_frame(blink_frame.copy(), landmarks=blink_landmarks)

        self.assertTrue(np.array_equal(output, blink_frame))
        self.assertIsNone(corrector._smoothed_correction)

    def test_gaze_lock_resets_camera_target_after_blink(self):
        frame, landmarks = self._scene()
        corrector = EyeContactCorrector()
        corrector.enabled = True
        corrector.mode = "gaze_lock"
        corrector.process_frame(frame.copy(), landmarks=landmarks)
        self.assertIsNotNone(corrector._locked_camera_target)

        blink_frame, blink_landmarks = self._scene(blink_eye=0)
        output = corrector.process_frame(
            blink_frame.copy(),
            landmarks=blink_landmarks,
        )

        self.assertTrue(np.array_equal(output, blink_frame))
        self.assertIsNone(corrector._locked_camera_target)

    def test_extreme_side_gaze_is_not_corrected(self):
        frame, landmarks = self._scene(left_offset=(20, 0), right_offset=(20, 0))
        corrector = EyeContactCorrector()
        corrector.enabled = True
        corrector.intensity = 1.0

        output = corrector.process_frame(frame.copy(), landmarks=landmarks)

        self.assertTrue(np.array_equal(output, frame))

    def test_side_gaze_correction_tapers_before_cutoff(self):
        moderate_frame, moderate_landmarks = self._scene(
            left_offset=(12, 0),
            right_offset=(12, 0),
        )
        edge_frame, edge_landmarks = self._scene(
            left_offset=(16, 0),
            right_offset=(16, 0),
        )

        moderate = EyeContactCorrector()
        moderate.enabled = True
        moderate.intensity = 1.0
        moderate_output = moderate.process_frame(
            moderate_frame.copy(),
            landmarks=moderate_landmarks,
        )

        edge = EyeContactCorrector()
        edge.enabled = True
        edge.intensity = 1.0
        edge_output = edge.process_frame(edge_frame.copy(), landmarks=edge_landmarks)

        moderate_before = self._iris_centers(moderate_frame)
        moderate_after = self._iris_centers(moderate_output)
        edge_before = self._iris_centers(edge_frame)
        edge_after = self._iris_centers(edge_output)
        moderate_motion = np.mean([
            before[0] - after[0]
            for before, after in zip(moderate_before, moderate_after)
        ])
        edge_motion = np.mean([
            before[0] - after[0]
            for before, after in zip(edge_before, edge_after)
        ])

        self.assertGreater(edge_motion, 1.0)
        self.assertLess(edge_motion, moderate_motion)

    def test_head_yaw_moves_shared_target_opposite_nose_offset(self):
        frame, frontal_landmarks = self._scene(left_offset=(0, 0), right_offset=(0, 0))
        _, yawed_landmarks = self._scene(
            left_offset=(0, 0),
            right_offset=(0, 0),
            nose_offset=12,
        )
        frontal = EyeContactCorrector()
        frontal.enabled = True
        frontal.intensity = 1.0
        yawed = EyeContactCorrector()
        yawed.enabled = True
        yawed.intensity = 1.0

        frontal_output = frontal.process_frame(frame.copy(), landmarks=frontal_landmarks)
        yawed_output = yawed.process_frame(frame.copy(), landmarks=yawed_landmarks)
        frontal_center = np.mean([point[0] for point in self._iris_centers(frontal_output)])
        yawed_center = np.mean([point[0] for point in self._iris_centers(yawed_output)])

        self.assertLess(yawed_center, frontal_center - 0.5)

    def test_consistent_pair_uses_low_latency_smoothing(self):
        centered_frame, centered_landmarks = self._scene(
            left_offset=(0, 0),
            right_offset=(0, 0),
        )
        gaze_frame, gaze_landmarks = self._scene()

        warmed = EyeContactCorrector()
        warmed.enabled = True
        warmed.intensity = 1.0
        warmed.process_frame(centered_frame.copy(), landmarks=centered_landmarks)
        warmed_output = warmed.process_frame(gaze_frame.copy(), landmarks=gaze_landmarks)

        fresh = EyeContactCorrector()
        fresh.enabled = True
        fresh.intensity = 1.0
        fresh_output = fresh.process_frame(gaze_frame.copy(), landmarks=gaze_landmarks)

        input_centers = self._iris_centers(gaze_frame)
        warmed_centers = self._iris_centers(warmed_output)
        fresh_centers = self._iris_centers(fresh_output)
        warmed_motion = np.mean([
            input_point[0] - output_point[0]
            for input_point, output_point in zip(input_centers, warmed_centers)
        ])
        fresh_motion = np.mean([
            input_point[0] - output_point[0]
            for input_point, output_point in zip(input_centers, fresh_centers)
        ])

        self.assertGreater(warmed_motion, fresh_motion * 0.70)

    def test_one_eye_jump_uses_stronger_smoothing(self):
        centered_frame, centered_landmarks = self._scene(
            left_offset=(0, 0),
            right_offset=(0, 0),
        )
        stable_frame, stable_landmarks = self._scene(
            left_offset=(8, 0),
            right_offset=(8, 0),
        )
        noisy_frame, noisy_landmarks = self._scene(
            left_offset=(12, 0),
            right_offset=(4, 0),
        )

        stable = EyeContactCorrector()
        stable.enabled = True
        stable.intensity = 1.0
        stable.process_frame(centered_frame.copy(), landmarks=centered_landmarks)
        stable.process_frame(stable_frame.copy(), landmarks=stable_landmarks)

        noisy = EyeContactCorrector()
        noisy.enabled = True
        noisy.intensity = 1.0
        noisy.process_frame(centered_frame.copy(), landmarks=centered_landmarks)
        noisy.process_frame(noisy_frame.copy(), landmarks=noisy_landmarks)

        self.assertGreater(
            abs(float(stable._smoothed_correction[0])),
            abs(float(noisy._smoothed_correction[0])) * 1.7,
        )

    def test_gaze_lock_reduces_small_coordinated_eye_motion(self):
        _, start_landmarks = self._scene(
            left_offset=(4, 0),
            right_offset=(4, 0),
        )
        _, moved_landmarks = self._scene(
            left_offset=(8, 0),
            right_offset=(8, 0),
        )

        corrected_motion = {}
        for mode in ("natural", "gaze_lock"):
            corrector = EyeContactCorrector()
            corrector.enabled = True
            corrector.intensity = 0.35
            corrector.mode = mode
            start_shifts = self._shifts(corrector, start_landmarks)
            moved_shifts = self._shifts(corrector, moved_landmarks)
            start_position = 4.0 + np.mean([shift[0] for shift in start_shifts])
            moved_position = 8.0 + np.mean([shift[0] for shift in moved_shifts])
            corrected_motion[mode] = abs(float(moved_position - start_position))

        self.assertLess(
            corrected_motion["gaze_lock"],
            corrected_motion["natural"] * 0.6,
        )

    def test_gaze_lock_is_stronger_at_default_intensity(self):
        _, landmarks = self._scene()
        mean_shift = {}
        for mode in ("natural", "gaze_lock"):
            corrector = EyeContactCorrector()
            corrector.enabled = True
            corrector.intensity = 0.35
            corrector.mode = mode
            shifts = self._shifts(corrector, landmarks)
            mean_shift[mode] = abs(float(np.mean([shift[0] for shift in shifts])))

        self.assertGreater(mean_shift["gaze_lock"], mean_shift["natural"] * 1.25)

    def test_gaze_lock_holds_small_head_jitter_and_follows_large_turn(self):
        _, frontal_landmarks = self._scene(
            left_offset=(0, 0),
            right_offset=(0, 0),
        )
        _, jitter_landmarks = self._scene(
            left_offset=(0, 0),
            right_offset=(0, 0),
            nose_offset=5,
        )
        _, turned_landmarks = self._scene(
            left_offset=(0, 0),
            right_offset=(0, 0),
            nose_offset=16,
        )
        corrector = EyeContactCorrector()
        corrector.enabled = True
        corrector.mode = "gaze_lock"

        self._shifts(corrector, frontal_landmarks)
        self._shifts(corrector, jitter_landmarks)
        self.assertAlmostEqual(float(corrector._locked_camera_target[0]), 0.0)

        self._shifts(corrector, turned_landmarks)
        first_turn_target = float(corrector._locked_camera_target[0])
        for _ in range(4):
            self._shifts(corrector, turned_landmarks)

        self.assertLess(first_turn_target, -0.015)
        self.assertLess(float(corrector._locked_camera_target[0]), first_turn_target)


if __name__ == "__main__":
    unittest.main()
