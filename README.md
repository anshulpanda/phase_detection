# phase-detection

Frame-level tennis forehand **swing-phase detection**: a Temporal Convolutional
Network over pose keypoints that labels every frame of a clip with one of seven
swing phases, plus the active-learning pipeline used to grow its training set
from 82 hand-labelled videos to 884.

```
no_phase → Start of Unit Turn → End of Backswing → Forward Swing Initiation
         → Acceleration → Contact → Follow Through
```

The headline metric is **boundary error in milliseconds** — how far off the
predicted start/end of each phase is from the labelled one. The final model
lands at a **16.7 ms median / 40.1 ms mean** on held-out players (roughly one
frame at 60 fps).

---

## How it works

**1. Pose → features.** A YOLO pose model gives 17 COCO keypoints per frame.
Those are normalized to be invariant to where the player is in frame and how
big they are: centered on the hip midpoint, scaled by torso length. 51 features
per frame (17 keypoints × [x, y, confidence]).

**2. TCN → per-frame probabilities.** A 4-block dilated TCN (dilations 1/2/4/8,
receptive field 121 frames ≈ 2.0 s at 60 fps, sized against the ~2 s maximum
swing duration) emits a probability distribution over the 7 phases *for each
frame*, independently.

**3. Monotonic decode → the actual labels.** Per-frame argmax flickers at
transitions and can produce physically impossible sequences. Instead a
Viterbi-style dynamic program finds the single highest-scoring *legal* path
across the whole clip, where legal means: stay in the current phase, advance
exactly one step, or (optionally) enter from `no_phase`. The TCN is per-frame
evidence; the decoder is the editor that stitches it into one coherent swing.

Training additionally uses **ordinal label smoothing** — phase transitions are
genuinely continuous, so a frame's target shares mass with its adjacent phases
rather than being one-hot — and **inverse-frequency class weighting**, since
`no_phase` + `Follow Through` are ~65% of frames while `Contact` is ~3%.

### The active-learning loop

The dataset was grown over three rounds, each: predict on unlabelled video →
score confidence → sample a stratified review set → human audit → fold the
results back into training.

Confidence is layered — per-frame (probability of the *decoded* class, not the
raw argmax), averaged per phase, then combined per video:

```
composite_confidence = 0.6 × min(phase confidences) + 0.4 × mean(phase confidences)
```

Weighted toward the weakest phase on purpose: a video shouldn't hide one broken
phase behind six good ones, because a single bad phase makes the whole labelled
swing unusable. Above a composite of 0.55 the round-1 audit found ~100% of
predictions correct, so above that threshold predictions are auto-accepted as
pseudo-labels; below it they go to a human.

| Round | Training videos | Test median | Test mean |
|-------|-----------------|-------------|-----------|
| 1     | 82 (ground truth) | 33.3 ms | 64.0 ms |
| 2     | 153 | 50.1 ms | 118.8 ms |
| 3 (final) | 884 | **16.7 ms** | **40.1 ms** |

Round 2 got *worse* before it got better: the pool it drew from included
slow-motion footage averaging ~10.4 s against ~2.4 s for normal speed — far
beyond the model's 2 s receptive field. Round 3 fixes this by time-resampling
slow-motion clips to the normal-speed mean duration (`--normalize-slowmo`),
preserving the whole swing rather than discarding the clip.

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

## Commands

Every command is a subcommand of `phase-detect`; `--help` works on each.

| Command | Does |
|---|---|
| `build-index` | chaptered videos → video index CSV (via ffprobe) |
| `pool-metadata` | probe the whole pool for fps / `is_slowmo` |
| `extract-pose` | video index → per-frame pose + label dataset |
| `merge-datasets` | previous + new round → combined training set |
| `train` | training set → checkpoint (+ curves) |
| `evaluate` | checkpoint → boundary-error / accuracy report |
| `predict` | checkpoint + videos → predictions CSV |
| `analyze-predictions` | predictions → missing-phase report |
| `build-review-set` | predictions → stratified human-review set |
| `resolve-review` | audit results → next round's video index |
| `export` | checkpoint → `.pt` and/or `.mlpackage` |

### Reproducing the final model

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

# …repeat. the final model:
phase-detect train --dataset joined_dataset_round3.csv \
                   --output best_phase_tcn_round3_slowmo_normalized.pt \
                   --split normal_speed_player_held_out \
                   --test-video-target 150 \
                   --normalize-slowmo

phase-detect export --model best_phase_tcn_round3_slowmo_normalized.pt \
                    --format both
```

### Evaluating

```bash
phase-detect evaluate \
  --model best_phase_tcn_round3_slowmo_normalized.pt \
  --dataset joined_dataset_round3.csv \
  --split normal_speed_player_held_out --test-video-target 150 \
  --normalize-slowmo
```

Reports pooled boundary error on **both** train and test, so the generalization
gap is visible in milliseconds rather than only in loss units, plus per-phase
and per-player breakdowns, decoded vs. raw accuracy, and a per-class report.

---

## Notes on the metrics

**Overall statistics pool raw measurements — they are not averages of the
per-phase statistics.** Medians and standard deviations don't average: a
combined median depends on the sorted order of every individual value, and
combined variance includes a between-group term that averaging per-phase
spreads throws away. Only a size-weighted mean would survive that shortcut, so
every start- and end-error is pooled into one flat distribution first.

**Missed phases are excluded from the error statistics.** A phase the decode
never emitted has no boundary to measure, so it's dropped rather than counted
as a large error — which flatters the numbers slightly. `n_missed` is reported
alongside so it stays visible.

**Two decode variants exist, and they give different numbers.** The
training/evaluation path allows the `no_phase` reset edge; the prediction path
does not, which guarantees every phase gets a contiguous span so that a missing
phase remains a meaningful signal for the review stage. `predict
--allow-no-phase-reset` switches to the training-style decode. Match the variant
to whatever you're comparing against.

**Checkpoint selection.** By default the best checkpoint is chosen on the test
split, matching the original experiments — which makes the reported test number
mildly optimistic. `train --val-from-train 0.15` carves a validation slice out
of training instead, so selection never touches test.

---

## Architecture

The package mirrors the pipeline-stage layout of the data-generating project
(`pose_estimation/`, `training_pipeline/{generate_dataset,train,predict}`,
`review_pipeline/` in `phase_model`) so a module's folder tells you which
stage of the pipeline it belongs to:

```
src/phase_detection/
├── constants.py            phase vocabulary, keypoint layout, label typo fixes
├── paths.py                data-root resolution (env var / .env / --data-root)
├── decode.py                monotonic Viterbi decode, boundary + confidence extraction
│                             (shared: used at both train-time evaluation and inference)
├── cli.py                   argparse subcommands, the single entry point
│
├── pose_estimation/
│   └── pose.py               YOLO wrapper, ffprobe metadata, video discovery
│                              (shared: used by both dataset generation and inference)
│
├── generate_dataset/
│   ├── indexing.py           video index construction
│   ├── extraction.py         pose + label extraction, dataset merging
│   ├── features.py           pose normalization, missed-detection handling, resampling
│   └── dataset.py            dataset loading, sequence building, padding/collation
│
├── train/
│   ├── model.py               PhaseTCN, checkpoint save/load, device resolution
│   ├── losses.py              ordinal label smoothing, class weights, masked soft CE
│   ├── splits.py              the three train/test split strategies
│   ├── train.py               training loop
│   ├── evaluate.py            boundary-error metrics and reports
│   └── export.py              TorchScript + Core ML export
│
├── predict/
│   └── predict.py             inference pipeline
│
└── review_pipeline/
    ├── confidence.py          composite confidence, banding, stratified sampling
    └── review.py              audit → training-label resolution
```

`constants.py`, `paths.py`, and `decode.py` stay at the top level rather than
inside a stage folder because they're genuinely cross-cutting: `decode.py`,
for instance, is used by both `train/evaluate.py` and `predict/predict.py`, so
it doesn't belong to either one. `pose_estimation/pose.py` gets its own
top-level folder for the same reason — `generate_dataset/extraction.py` and
`predict/predict.py` both depend on it — mirroring how `phase_model` keeps
`pose_estimation/` as a sibling of `training_pipeline/` rather than nesting it
inside.

Run the tests with `pytest` — they cover the decode's transition grammar,
normalization invariances, the smoothing distribution, the confidence formula,
and split disjointness, and need no data or checkpoints.

## Deploying the exported model

`export` emits softmax probabilities, **not** decoded phases — the monotonic
decode is a sequence-level dynamic program that traces poorly, so the client
applies it. A consumer must therefore: normalize keypoints identically (hip
midpoint, torso scale), run the model, then run the decode over the returned
probabilities. The phase order is stored in the `.mlpackage` metadata under
`phase_order`.
