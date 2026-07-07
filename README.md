# violiese3

## OpenVINS comparison experiments

The aligned pose-only OpenVINS dataset can be evaluated with:

```bash
python openvins_experiments.py
```

The runner prints every methodological change and its reason, then reports:

- joint 6D SE(3) coverage;
- a block-diagonal SE(3) covariance ablation;
- translational and rotational marginal coverage;
- tangent-space and projected translation volumes;
- coverage-per-volume efficiency;
- a Euclidean position-sphere conformal baseline.

It evaluates three leakage-safe protocols:

- in-distribution chronological blocks from the same sequence;
- cross-sequence testing on an unseen sequence from the same environment;
- cross-environment testing on an unseen EuRoC environment family.

The uploaded comparison CSVs do not contain VIO covariance. Consequently, this
runner fits a full empirical 6x6 covariance on a separate fit split. The
covariance-aware per-pose method remains in `minimal_run.py`.

The canonical residual convention is `BODY`:

```text
xi = Log(T_est^{-1} T_gt)
```

## Unified pipeline

Use the same calibration and projection pipeline with either pose-only datasets:

```bash
python run_pipeline.py --dataset-dir ../datasets/OPENVINS
```

or legacy merged VIO files that contain per-pose covariance:

```bash
python run_pipeline.py --merged-csv data/VIO/V1_01_easy_merged.csv
```

The default merged-data alignment is a Horn rigid transform fitted on the first
20% of the trajectory and then frozen. Compare it against full-trajectory
benchmark alignment with:

```bash
python alignment_experiment.py
```

This also generates full test-trajectory figures for `V2_02_medium_merged` with
seven calibrated task-space regions. Green stars are jointly covered GT poses;
red stars are misses. Select another sequence or task point with:

```bash
python alignment_experiment.py \
  --visualize-sequence V2_02_medium_merged \
  --body-point 1,0,0 \
  --region-count 8
```

Both paths use the shared implementation in
`utils/conformal_prediction/se3.py`, with dataset adapters in
`utils/dataset_io.py`. Pose-only data use a clearly labeled empirical
covariance fallback. Merged data use dynamic per-pose VIO covariance. Legacy
merged files are aligned together with their world-frame covariance, which is
then converted into the body tangent frame.
Use `--covariance-frame body` only when the source covariance is already in
body coordinates. `--alignment-mode full_trajectory` uses all ground-truth
poses and is only a benchmark comparison, while `initial_block` is the causal
CP experiment. The generated 3D plot projects the calibrated SE(3)
set onto the camera origin and a forward body-attached point; the latter makes
rotation-induced curvature visible.

The `data/VIO` files are retained only for the merged-covariance and alignment
fallback path. The current OpenVINS covariance experiments use the aligned
OpenVINS export loaded by `openvins_covariance_experiment.py`,
`adaptive_openvins_experiment.py`, and `openvins_volume_reduction_experiment.py`.

## Aligned OpenVINS covariance comparison

The `datasets/OPEN_VINS` export is already EVO-aligned. All eleven sequences
contain translation and rotation covariance and support the full comparison:

```bash
python openvins_covariance_experiment.py
```

The methods are OpenVINS covariance, full empirical 6D covariance, independent
translation/rotation blocks, and independent per-axis scales. No additional
trajectory alignment is applied. The default covariance convention is
`position_world_rotation_body`, matching OpenVINS' additive global position
error and left quaternion orientation error. Other frame choices remain
available as explicit ablations.

The complete-trajectory ACI/AgACI experiment now uses full dynamic OpenVINS
`Pt+Pr` covariance for every sequence:

```bash
python adaptive_openvins_experiment.py
```

It evaluates same-environment, pooled leave-one-trajectory-out, and
cross-environment protocols without splitting a trajectory across roles.

## OpenVINS volume-reduction ablations

Run the 90% coverage-volume study with:

```bash
python openvins_volume_reduction_experiment.py
```

The runner reuses the pooled and leave-one-environment-out trajectory folds
from `adaptive_openvins_experiment.py`. It compares OpenVINS correlation
shrinkage, fit-only standardized-residual corrections, dynamic empirical
blends, covariance-conditioned score scaling, and causal recent-score ACI and
AgACI variants. Models that alter covariance shape or predict a score scale are
fitted only on complete fit trajectories; conformal quantiles use separate
calibration trajectories.

Global scalar covariance inflation is intentionally omitted because it cannot
change a conformally calibrated set: scaling covariance by `c` divides the
Mahalanobis radius by `sqrt(c)`, exactly cancelling in the final ellipsoid.
Recent-score and all ACI/AgACI updates require the pose error to become
available after prediction. On a robot this requires delayed supervision such
as GPS, motion capture, loop-closure/map localization, or another trusted pose
reference; without it, only the frozen split-conformal variant is deployable.
