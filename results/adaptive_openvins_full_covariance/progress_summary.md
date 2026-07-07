# Full-Covariance Adaptive OpenVINS Summary

Date: 2026-06-27

The updated dataset supplies aligned poses and dynamic OpenVINS translation
and rotation covariance for all eleven trajectories. Complete trajectories are
kept disjoint across fit, calibration, and test roles. Values below use a 90%
target and report trajectory-mean coverage / task-space volume in cubic metres.

| Protocol | Covariance | Method | Coverage | Volume |
|---|---|---:|---:|---:|
| Same environment | Empirical full 6D | ACI | 0.762 | 0.0791 |
| Same environment | Empirical full 6D | AgACI-EWA | 0.786 | 0.0621 |
| Same environment | OpenVINS full 6D | ACI | 0.806 | 19.90 |
| Same environment | OpenVINS full 6D | AgACI-EWA | 0.808 | 12.41 |
| Pooled same robot | Empirical full 6D | ACI | 0.904 | 0.0492 |
| Pooled same robot | Empirical full 6D | AgACI-EWA | 0.933 | 0.0389 |
| Pooled same robot | OpenVINS full 6D | ACI | 0.898 | 19.37 |
| Pooled same robot | OpenVINS full 6D | AgACI-EWA | 0.910 | 16.94 |
| Cross environment | Empirical full 6D | ACI | 0.801 | 0.0628 |
| Cross environment | Empirical full 6D | AgACI-EWA | 0.824 | 0.0533 |
| Cross environment | OpenVINS full 6D | ACI | 0.898 | 20.98 |
| Cross environment | OpenVINS full 6D | AgACI-EWA | 0.904 | 16.99 |

The full OpenVINS covariance preserves near-target pooled and cross-environment
coverage while reducing the old Pt-plus-empirical-rotation hybrid volume from
roughly 255-652 m^3 to roughly 17-21 m^3. The empirical covariance remains much
more compact, but undercovers under cross-environment transfer.

AgACI-EWA is the radial exponential-weights implementation in this repository,
not the exact BOA aggregation from the original AgACI paper.
