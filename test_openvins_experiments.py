import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from adaptive_openvins_experiment import (
    build_cross_environment_folds,
    build_pooled_same_robot_folds,
    build_same_environment_folds,
    environment_from_name,
    run_aci,
)
from utils.dataset_io import estimate_horn_alignment
from utils.conformal_prediction.se3 import (
    conformal_quantile,
    project_pose_set_to_task_space,
    se3_body_error,
)


class OpenVINSExperimentTests(unittest.TestCase):
    def test_complete_trajectory_folds_are_disjoint(self):
        sequences = {
            "OV_V1_01_easy": None,
            "OV_V1_02_medium": None,
            "OV_V1_03_difficult": None,
        }
        folds = build_same_environment_folds(sequences)
        self.assertEqual(len(folds), 3)
        for fold in folds:
            self.assertNotIn(fold.test_name, fold.fit_names)
            self.assertNotIn(fold.test_name, fold.calibration_names)
            self.assertTrue(set(fold.fit_names).isdisjoint(fold.calibration_names))

    def test_pooled_folds_use_multiple_environments_without_leakage(self):
        sequences = {
            f"OV_{environment}_{index:02d}_test": None
            for environment in ("MH", "V1", "V2")
            for index in range(1, 4)
        }
        folds = build_pooled_same_robot_folds(sequences)
        self.assertEqual(len(folds), len(sequences))
        for fold in folds:
            self.assertNotIn(fold.test_name, fold.fit_names)
            self.assertNotIn(fold.test_name, fold.calibration_names)
            self.assertEqual(
                {environment_from_name(name) for name in fold.calibration_names},
                {"MH", "V1", "V2"},
            )

    def test_cross_environment_folds_exclude_target_from_sources(self):
        sequences = {
            f"OV_{environment}_{index:02d}_test": None
            for environment in ("MH", "V1", "V2")
            for index in range(1, 4)
        }
        folds = build_cross_environment_folds(sequences)
        for fold in folds:
            source_names = (*fold.fit_names, *fold.calibration_names)
            self.assertTrue(
                all(
                    environment_from_name(name) != fold.test_environment
                    for name in source_names
                )
            )

    def test_aci_reduces_effective_alpha_after_miss(self):
        calibration_scores = np.arange(1.0, 101.0)
        result = run_aci(
            calibration_scores,
            np.array([1000.0, 0.0]),
            alpha=0.1,
            gamma=0.01,
        )
        self.assertFalse(result.covered[0])
        self.assertLess(result.effective_alphas[1], result.effective_alphas[0])

    def test_identity_pose_has_zero_error(self):
        pose = np.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]])
        np.testing.assert_allclose(se3_body_error(pose, pose), 0.0, atol=1e-12)

    def test_body_translation_rotates_with_estimated_frame(self):
        quaternion = Rotation.from_euler("z", 90, degrees=True).as_quat()
        estimated = np.array([[0.0, 0.0, 0.0, *quaternion]])
        ground_truth = np.array([[1.0, 0.0, 0.0, *quaternion]])
        error = se3_body_error(estimated, ground_truth)
        np.testing.assert_allclose(error[0, :3], [0.0, -1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(error[0, 3:], 0.0, atol=1e-12)

    def test_finite_sample_quantile_uses_conformal_rank(self):
        scores = np.arange(1.0, 10.0)
        self.assertEqual(conformal_quantile(scores, alpha=0.2), 8.0)

    def test_body_error_is_invariant_to_common_world_transform(self):
        estimated = np.array([[0.2, -0.3, 1.1, 0.0, 0.0, 0.0, 1.0]])
        ground_truth = np.array(
            [[0.4, -0.1, 1.0, *Rotation.from_euler("y", 10, degrees=True).as_quat()]]
        )
        world_rotation = Rotation.from_euler("xyz", [20, -5, 30], degrees=True)
        world_translation = np.array([2.0, -1.0, 0.5])

        def transform(pose):
            return np.column_stack(
                [
                    world_rotation.apply(pose[:, :3]) + world_translation,
                    (world_rotation * Rotation.from_quat(pose[:, 3:])).as_quat(),
                ]
            )

        original = se3_body_error(estimated, ground_truth)
        transformed = se3_body_error(transform(estimated), transform(ground_truth))
        np.testing.assert_allclose(original, transformed, atol=1e-12)

    def test_rotation_moves_nonzero_task_point_but_not_body_origin(self):
        estimated = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        tangent_samples = np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2.0]]
        )
        projected = project_pose_set_to_task_space(
            estimated,
            tangent_samples,
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        np.testing.assert_allclose(projected[0, 0], [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(projected[0, 1], [0.0, 1.0, 0.0], atol=1e-12)

    def test_horn_alignment_recovers_metric_rigid_transform(self):
        source = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        )
        expected_rotation = Rotation.from_euler(
            "xyz", [12.0, -8.0, 35.0], degrees=True
        ).as_matrix()
        expected_translation = np.array([2.0, -1.0, 0.5])
        target = source @ expected_rotation.T + expected_translation
        rotation, translation = estimate_horn_alignment(source, target)
        np.testing.assert_allclose(rotation, expected_rotation, atol=1e-12)
        np.testing.assert_allclose(translation, expected_translation, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
