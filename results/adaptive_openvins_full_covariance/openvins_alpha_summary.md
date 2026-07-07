# OpenVINS ACI/AgACI Alpha Summary

All results use full dynamic OpenVINS `Pt+Pr` covariance and complete-trajectory
fit/calibration/test folds. Each entry is trajectory-mean coverage / geometric
mean task-space volume in cubic metres for body point `[3, 0, 0]` m.

| Target | Pooled ACI | Pooled AgACI-EWA | Cross-env ACI | Cross-env AgACI-EWA |
|---:|---:|---:|---:|---:|
| 99% | 0.989 / 100.68 | 0.985 / 51.60 | 0.987 / 110.94 | 0.981 / 54.78 |
| 95% | 0.948 / 35.76 | 0.951 / 20.80 | 0.948 / 37.37 | 0.946 / 20.19 |
| 90% | 0.898 / 19.37 | 0.910 / 16.94 | 0.898 / 20.98 | 0.904 / 16.99 |
| 80% | 0.797 / 14.43 | 0.828 / 14.11 | 0.799 / 14.70 | 0.825 / 14.43 |
| 60% | 0.597 / 11.43 | 0.621 / 11.66 | 0.599 / 11.53 | 0.625 / 11.65 |
| 50% | 0.497 / 10.54 | 0.496 / 10.70 | 0.499 / 10.53 | 0.497 / 10.69 |

## Fixed-Pose Shape Check

The fixed sample is `OV_V1_02_medium`, pose 816, under the pooled-same-robot
protocol. Direct nonlinear SE(3) point clouds were used; no convex hull was
applied. The `[3, 0, 0]` m task-point regions are rounded anisotropic blobs,
not clearly banana-shaped. A perpendicular `[0, 3, 0]` m diagnostic and a
`[10, 0, 0]` m lever-arm diagnostic produced the same qualitative result.

This indicates that the complete conformal set, including translational and
multi-axis rotational uncertainty, fills the curved support. A visible banana
would require a different task geometry or stronger translation-rotation
cross-correlation; the exported covariance has zero cross blocks.

AgACI radii are adapted independently for each alpha and therefore are not
guaranteed to be nested or monotone across target levels.
