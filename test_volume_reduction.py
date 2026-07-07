import unittest

import numpy as np

from adaptive_openvins_experiment import task_log_volumes
from openvins_volume_reduction_experiment import (
    _correlation_shrinkage,
    fit_log_scale_model,
    predict_log_scale,
    run_recent_aci,
)
from utils.conformal_prediction.se3 import mahalanobis_scores


class VolumeReductionTests(unittest.TestCase):
    def test_scalar_covariance_rescaling_cancels_after_calibration(self) -> None:
        covariance = np.diag([0.2, 0.3, 0.4, 0.01, 0.02, 0.03])[None]
        radius = np.array([4.0])
        scalar = 25.0
        original = task_log_volumes(covariance, radius, np.array([3.0, 0.0, 0.0]))
        rescaled = task_log_volumes(
            scalar * covariance,
            radius / np.sqrt(scalar),
            np.array([3.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(original, rescaled, atol=1e-10)

    def test_correlation_shrinkage_preserves_variances(self) -> None:
        covariance = np.array(
            [[[2.0, 0.5], [0.5, 1.0]]], dtype=np.float64
        )
        shrunk = _correlation_shrinkage(covariance, 0.75)
        np.testing.assert_allclose(
            np.diagonal(shrunk, axis1=1, axis2=2), [[2.0, 1.0]]
        )
        self.assertAlmostEqual(shrunk[0, 0, 1], 0.125)

    def test_scale_model_uses_only_passed_fit_values(self) -> None:
        covariance = np.repeat(np.eye(6)[None], 20, axis=0)
        covariance[:, 0, 0] = np.linspace(1.0, 3.0, 20)
        errors = np.ones((20, 6))
        scores = mahalanobis_scores(errors, covariance)
        elapsed = np.arange(20, dtype=np.float64)
        model = fit_log_scale_model(
            covariance, elapsed, scores, include_time=False
        )
        prediction = predict_log_scale(model, covariance, elapsed)
        self.assertEqual(prediction.shape, (20,))
        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertTrue(np.all(prediction > 0.0))

    def test_recent_aci_is_causal(self) -> None:
        calibration = np.arange(1.0, 101.0)
        prefix = np.linspace(20.0, 80.0, 75)
        first = run_recent_aci(
            calibration, np.concatenate([prefix, [1.0]]), 0.1, 0.01, 100
        )
        second = run_recent_aci(
            calibration, np.concatenate([prefix, [1000.0]]), 0.1, 0.01, 100
        )
        np.testing.assert_allclose(first.radii[:76], second.radii[:76])


if __name__ == "__main__":
    unittest.main()
