# phase-detection

Frame-level tennis forehand **swing-phase detection**: a pose-based Temporal
Convolutional Network that labels every frame of a clip with one of seven
swing phases, plus the active-learning pipeline used to grow its training set
from 82 hand-labelled videos to 884.

```
no_phase → Start of Unit Turn → End of Backswing → Forward Swing Initiation
         → Acceleration → Contact → Follow Through
```

Headline metric is **boundary error in milliseconds** (how far off the
predicted phase start/end is from the labelled one): **16.7 ms median / 40.1
ms mean** on held-out players, the final model's result.

---

## Install

```bash
git clone <this repo> && cd phase_detection
pip install -e .              # add '[coreml]' for .mlpackage export (macOS only)
```

**Data lives outside this repo.** Point the code at it with any of:

```bash
export PHASE_DATA_ROOT=/path/to/phase_model     # environment variable
cp .env.example .env                            # or a .env file
phase-detect train --data-root /path/to/data …  # or per-command
```

The data root is expected to contain:

```
<data root>/
├── all/                                    # unlabelled video pool
├── ground_truth/                           # hand-chaptered videos
├── pose_estimation/yolo26n-pose.pt         # YOLO pose weights
├── video_pool_metadata.csv                 # fps / is_slowmo per video
└── training_pipeline/
    ├── generate_dataset/                   # dataset CSVs
    └── train/                              # checkpoints
```

---

## Where to go for what

Every command below is `phase-detect <subcommand>`; `--help` works on each.
The package folder tells you which stage owns it:

| I want to... | Go to | Commands |
|---|---|---|
| Build/extend a training set from video | [`generate_dataset/`](phase_detection/generate_dataset/) | `build-index`, `pool-metadata`, `extract-pose`, `merge-datasets` |
| Train or evaluate the model | [`train/`](phase_detection/train/) | `train`, `evaluate`, `export` |
| Run the model on new videos | [`predict/`](phase_detection/predict/) | `predict` |
| Score/sample predictions for human review | [`review_pipeline/`](phase_detection/review_pipeline/) | `analyze-predictions`, `build-review-set`, `resolve-review` |
| Pose extraction (shared by the above) | [`pose_estimation/`](phase_detection/pose_estimation/) | used internally by `extract-pose` and `predict` |

`constants.py`, `paths.py`, and `decode.py` sit at the package root because
they're used across more than one of those stages.

---

## Reproducing the final model

```bash
# 0. index + extract the hand-labelled ground truth
phase-detect build-index --output video_data.csv
phase-detect extract-pose --index video_data.csv --output joined_dataset.csv

# 1. train, predict on everything unlabelled, review the weakest predictions
phase-detect train   --dataset joined_dataset.csv --output best_phase_tcn.pt
phase-detect predict --model best_phase_tcn.pt \
                     --exclude-ids-csv joined_dataset.csv \
                     --output round1_predictions.csv
phase-detect build-review-set --predictions round1_predictions.csv \
                              --output review_set_round1.csv

# 2. …human audits review_set_round1.csv, correcting chapters in a video editor…

# 3. fold the audit back in and retrain
phase-detect resolve-review --audit review_results_round1.csv \
                            --predictions round1_predictions.csv \
                            --corrections /path/to/corrected_exports \
                            --threshold 0.55 \
                            --output video_data_round2_new.csv
phase-detect extract-pose --index video_data_round2_new.csv \
                          --output joined_round2_new.csv
phase-detect merge-datasets --previous joined_dataset.csv \
                            --new joined_round2_new.csv \
                            --output joined_dataset_round2.csv \
                            --source-tag round2

# …repeat rounds 2 and 3 the same way. the final model:
phase-detect train --dataset joined_dataset_round3.csv \
                   --output phase_detection.pt \
                   --split normal_speed_player_held_out \
                   --test-video-target 150 \
                   --normalize-slowmo

phase-detect export --model phase_detection.pt --format both
```

```bash
phase-detect evaluate \
  --model phase_detection.pt \
  --dataset joined_dataset_round3.csv \
  --split normal_speed_player_held_out --test-video-target 150 \
  --normalize-slowmo
```

| Round | Training videos | Test median | Test mean |
|-------|-----------------|-------------|-----------|
| 1     | 82 (ground truth) | 33.3 ms | 64.0 ms |
| 2     | 153 | 50.1 ms | 118.8 ms |
| 3 (final) | 884 | **16.7 ms** | **40.1 ms** |

---

## More detail

The *why*, not just the *how*, lives next to the code rather than here — read
the module docstring for the thing you're touching:

- **How the model works** (TCN architecture, monotonic Viterbi decode,
  ordinal label smoothing): `train/model.py`, `decode.py`, `train/losses.py`
- **Confidence scoring and the active-learning loop**: `review_pipeline/confidence.py`, `review_pipeline/review.py`
- **Metric definitions and pooling caveats**: `train/evaluate.py`
- **Deploying the exported model**: `train/export.py` (exports softmax
  probabilities, not decoded phases — the decode is a client-side step)

Run `pytest` for the invariants those modules guarantee (decode transition
grammar, normalization invariances, split disjointness, the confidence
formula) — no data or checkpoints required.
