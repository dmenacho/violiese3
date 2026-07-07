# OpenVINS quantile-volume study

Target coverage is 90%. A configuration is screened as coverage-valid only if
both pooled and cross-environment sample coverage are at least 89%.

## Recommended checkpoint

`correlation_shrink_0.75` + `loglinear_covariance` + `AgACI-EWA` with `calibration_only` feedback is the smallest
configuration that reaches at least 90% sample-pooled coverage in both protocols.

- Pooled: coverage 0.9051, median volume 1.2698 m^3.
- Cross-environment: coverage 0.9005, median volume 1.7587 m^3.
- Relative to raw OpenVINS AgACI, median volume falls by 92.5% pooled and 89.6% cross-environment.
- Worst individual trajectory coverage is 0.8086; aggregate validity does not imply uniform trajectory-level validity.

The method retains each OpenVINS marginal variance, shrinks its reported
off-diagonal correlations 75% toward zero, and predicts a pose-specific score
scale from covariance eigenvalue/diagonal features using fit trajectories.
AgACI then calibrates the normalized score on disjoint trajectories.

## Coverage-valid configurations

| covariance_strategy              | score_scale               | method    | feedback         |   pooled_coverage |   cross_environment_coverage |   worst_trajectory_coverage |   pooled_volume_m3 |   cross_environment_volume_m3 |   worst_rolling_mae |
|:---------------------------------|:--------------------------|:----------|:-----------------|------------------:|-----------------------------:|----------------------------:|-------------------:|------------------------------:|--------------------:|
| diagonal_openvins                | constant                  | AgACI-EWA | calibration_only |            0.8941 |                       0.8928 |                      0.7855 |             1.4937 |                        1.4778 |              0.0503 |
| correlation_shrink_0.75          | constant                  | AgACI-EWA | calibration_only |            0.8956 |                       0.8929 |                      0.7797 |             1.5290 |                        1.5055 |              0.0505 |
| correlation_shrink_0.50          | constant                  | AgACI-EWA | calibration_only |            0.8944 |                       0.8923 |                      0.7758 |             1.6648 |                        1.6682 |              0.0527 |
| correlation_shrink_0.75          | loglinear_covariance      | AgACI-EWA | calibration_only |            0.9051 |                       0.9005 |                      0.8086 |             1.2698 |                        1.7587 |              0.0585 |
| correlation_shrink_0.50          | loglinear_covariance      | AgACI-EWA | calibration_only |            0.8937 |                       0.9151 |                      0.7420 |             1.3702 |                        1.9766 |              0.0714 |
| correlation_shrink_0.25          | constant                  | AgACI-EWA | calibration_only |            0.8944 |                       0.8919 |                      0.7676 |             2.0388 |                        2.0310 |              0.0529 |
| correlation_shrink_0.50          | loglinear_covariance_time | AgACI-EWA | calibration_only |            0.9040 |                       0.9122 |                      0.7710 |             1.5455 |                        2.1528 |              0.0600 |
| diagonal_openvins_fit_bias       | constant                  | AgACI-EWA | calibration_only |            0.8973 |                       0.8919 |                      0.7957 |             1.7730 |                        2.5313 |              0.0491 |
| correlation_shrink_0.75_fit_bias | constant                  | AgACI-EWA | calibration_only |            0.8971 |                       0.8903 |                      0.7966 |             1.8143 |                        2.5596 |              0.0488 |
| correlation_shrink_0.25          | loglinear_covariance_time | AgACI-EWA | calibration_only |            0.9011 |                       0.9175 |                      0.7836 |             1.9968 |                        2.6777 |              0.0591 |
| correlation_shrink_0.25          | loglinear_covariance      | AgACI-EWA | calibration_only |            0.8941 |                       0.9059 |                      0.7420 |             1.8577 |                        2.6941 |              0.0676 |
| diagonal_openvins                | constant                  | ACI       | calibration_only |            0.8941 |                       0.8911 |                      0.7560 |             2.6979 |                        2.6271 |              0.0857 |
| correlation_shrink_0.75          | constant                  | ACI       | calibration_only |            0.8938 |                       0.8910 |                      0.7512 |             2.7383 |                        2.7082 |              0.0864 |
| diagonal_openvins_fit_bias       | loglinear_covariance      | AgACI-EWA | calibration_only |            0.8952 |                       0.8986 |                      0.7177 |             1.6722 |                        2.8225 |              0.0568 |
| diagonal_openvins_fit_bias       | loglinear_covariance_time | AgACI-EWA | calibration_only |            0.8921 |                       0.8986 |                      0.6899 |             1.5774 |                        2.8604 |              0.0589 |
| correlation_shrink_0.50_fit_bias | constant                  | AgACI-EWA | calibration_only |            0.8968 |                       0.8915 |                      0.8106 |             2.0214 |                        2.9196 |              0.0490 |
| correlation_shrink_0.50          | constant                  | ACI       | calibration_only |            0.8938 |                       0.8910 |                      0.7449 |             2.9925 |                        2.9779 |              0.0884 |
| correlation_shrink_0.75_fit_bias | loglinear_covariance      | AgACI-EWA | calibration_only |            0.9041 |                       0.9135 |                      0.8089 |             1.7317 |                        3.0310 |              0.0506 |
| correlation_shrink_0.75_fit_bias | loglinear_covariance_time | AgACI-EWA | calibration_only |            0.9043 |                       0.9070 |                      0.8115 |             1.6750 |                        3.1545 |              0.0494 |
| correlation_shrink_0.50_fit_bias | loglinear_covariance      | AgACI-EWA | calibration_only |            0.8919 |                       0.9165 |                      0.6957 |             1.7362 |                        3.6599 |              0.0617 |

Scalar covariance inflation is intentionally absent: conformal calibration
cancels any global scalar and cannot change the final calibrated set.

Recent-score variants require delayed ground-truth/error feedback and are not
deployable on an unlabeled robot without an external localization signal. In
this experiment, 100/300-sample rolling replacements undercovered and are rejected.
Empirical covariance blends reached very small volumes but failed cross-environment
coverage, so they are also rejected.
