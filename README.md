<p align="center"><img src=".github/icon.svg" width="96" height="96" alt="Theia app icon"></p>

<h1 align="center">Theia</h1>

<p align="center"><b>Stay strong, stay safe: a camera coach against injury and bone loss. On Earth and in space.</b></p>

<p align="center">
  <a href="https://theia-app.pages.dev">Live demo</a> ·
  <a href="https://theia-app.pages.dev/space">Space</a> ·
  <a href="https://theia-app.pages.dev/earth">Earth</a>
</p>

## Pose-based movement analysis

Research code for extracting body poses from RGB video, measuring movement, learning unusual gait patterns, and visualizing the body regions involved. The project explores walking, running, ground squats, and astronaut exercise footage.

## Repository contents

- `scripts/`: video preparation, pose extraction, quality checks, gait features, model training, evaluation, and HTML report/demo builders.
- `scripts/gait_distribution_demo/`: browser visualization source for distribution-based movement analysis.
- `scripts/gait_assessment_report/`: assessment report interface.
- `tests/`: pipeline, checkpoint, archive, metrics, and playback checks.

This is a **code-only research repository**. Videos, participant data, extracted poses, annotations, trained weights, third-party code, generated reports, papers, and offline demo archives are excluded by `.gitignore`. A fresh clone does not contain the inputs needed to reproduce the local experiments or play the populated demo.

## Model and experiment entry points

| Component | Source |
| --- | --- |
| Sequence model | `scripts/gait_from_scratch_model.py` |
| Geometric pose features | `scripts/gait_kinematic_model.py` |
| Statistical pooling model | `scripts/gait_statistical_pool_model.py` |
| Dataset and window preparation | `scripts/gait_assessment_data.py` |
| Training and evaluation helpers | `scripts/gait_from_scratch_training.py`, `scripts/gait_from_scratch_evaluation.py` |
| Comparison experiments | `scripts/run_gait_improvement.py`, `scripts/run_gait_improvement_refinement.py` |
| Evaluation and reporting | `scripts/evaluate_gait_improvement.py`, `scripts/report_gait_improvement.py` |
| Demo prediction refresh | `scripts/update_gait_demo_pooling.py` |
| Offline demo packaging | `scripts/package_gait_selected_offline.py` |

The statistical pooling model combines learned frame features with mean, standard deviation, minimum, and maximum pooling. Models consume pose sequences and declared view/environment context; amputation-group labels are not model inputs.

## Running the code

Use Python 3 and run scripts from the repository root. Individual tools have different dependencies and input requirements; inspect their arguments and path constants before running them. Training uses PyTorch and NumPy. Pose extraction additionally requires the upstream Sapiens2 implementation, checkpoint, and a compatible CUDA GPU. Video preparation uses FFmpeg; browser components use HTML, CSS, and JavaScript.

`scripts/setup_pose_environment.sh` records the pinned local pose-extraction setup, including a roughly 20 GB Sapiens2 checkpoint download. Review its environment and storage requirements before running it. There is not yet a unified installation or complete reproducibility package. Several experiment/report scripts reference dated local manifests and generated artifacts that must be supplied separately.

A model-only structural check, with PyTorch installed:

```bash
python3 scripts/gait_statistical_pool_model.py
```

This checks permutation and translation/scale invariance on synthetic input; it is not an accuracy evaluation. Some pipeline tests also require the excluded local manifests and fixtures.

## Interpretation

Current experimental targets are automated pose-derived reference rules, not independently adjudicated clinical diagnoses. An unusual movement score does not establish poor technique, injury, or a need for treatment. Projection, occlusion, pose quality, and camera view affect measurements. Internal and selected-example results should not be interpreted as external clinical validation.

## Authors

Oleksandr Namchuk · Anna-Oleksandra Danchenko · Hlib Plakhtii · Mariia Shevtsova

Third-party datasets, models, and media retain their own access and licensing conditions and are not redistributed here.
