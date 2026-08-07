"""
Command-line interface.

Subcommands follow the pipeline order:

    build-index          chaptered videos      -> video index CSV
    pool-metadata        whole video pool      -> fps/is_slowmo lookup
    extract-pose         video index           -> per-frame pose+label dataset
    merge-datasets       previous + new        -> combined training set
    train                training set          -> checkpoint (+ curves)
    evaluate             checkpoint            -> boundary-error report
    predict              checkpoint + videos   -> predictions CSV
    analyze-predictions  predictions           -> missing-phase report
    build-review-set     predictions           -> stratified human-review set
    resolve-review       audit results         -> next round's video index
    export               checkpoint            -> .pt / .mlpackage

Every subcommand accepts --data-root to point at the data directory; it also
reads PHASE_DATA_ROOT or a .env file. See README.md.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from .paths import DataPaths


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-root", default=None,
                   help="Path to the data directory (default: $PHASE_DATA_ROOT, .env, or ../phase_model)")


# ── build-index ────────────────────────────────────────────────────────────
def cmd_build_index(args) -> int:
    from .generate_dataset.indexing import build_ground_truth_index

    paths = DataPaths(args.data_root)
    video_dir = Path(args.videos) if args.videos else paths.ground_truth_dir
    df = build_ground_truth_index(
        video_dir, paths.root, exclude_slowmo=not args.include_slowmo
    )
    out = paths.dataset(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
    return 0


# ── pool-metadata ──────────────────────────────────────────────────────────
def cmd_pool_metadata(args) -> int:
    from .generate_dataset.indexing import build_pool_metadata

    paths = DataPaths(args.data_root)
    video_dir = Path(args.videos) if args.videos else paths.video_dir
    df = build_pool_metadata(video_dir, paths.root)
    out = Path(args.output) if args.output else paths.pool_metadata_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n{len(df)} videos, {int(df['is_slowmo'].sum())} slow-motion")
    print(f"Saved -> {out}")
    return 0


# ── extract-pose ───────────────────────────────────────────────────────────
def cmd_extract_pose(args) -> int:
    from .generate_dataset.extraction import extract_dataset
    from .train.model import resolve_yolo_device
    from .pose_estimation.pose import load_pose_model

    paths = DataPaths(args.data_root)
    index_csv = paths.dataset(args.index)
    video_index = pd.read_csv(index_csv)
    if args.limit:
        video_index = video_index.head(args.limit)

    yolo_device = resolve_yolo_device(args.device)
    yolo_path = Path(args.yolo_model) if args.yolo_model else paths.yolo_pose_model
    print(f"Pose model: {yolo_path}  (device: {yolo_device})")
    pose_model = load_pose_model(yolo_path, yolo_device)

    dataset = extract_dataset(video_index, paths.root, pose_model)

    out = paths.dataset(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out, index=False)
    print(f"\nSaved  -> {out}")
    print(f"Rows   : {len(dataset):,}")
    print(f"Videos : {dataset['video_id'].nunique()}")
    print(f"Labels :\n{dataset['phase_label'].value_counts().to_string()}")
    return 0


# ── merge-datasets ─────────────────────────────────────────────────────────
def cmd_merge_datasets(args) -> int:
    from .generate_dataset.extraction import merge_datasets

    paths = DataPaths(args.data_root)
    merge_datasets(
        paths.dataset(args.previous),
        paths.dataset(args.new),
        paths.dataset(args.output),
        source_tag=args.source_tag,
    )
    return 0


# ── train ──────────────────────────────────────────────────────────────────
def cmd_train(args) -> int:
    import numpy as np

    from .generate_dataset.dataset import build_sequences, load_and_clean, load_slowmo_map
    from .generate_dataset.features import normalize_slowmo_durations
    from .train.model import build_model, load_checkpoint, resolve_device
    from .train.splits import (
        normal_speed_player_held_out_split,
        player_held_out_split,
        random_video_split,
        summarize_split,
    )
    from .train.train import TrainConfig, plot_history, train_model

    paths = DataPaths(args.data_root)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    csv_path = paths.dataset(args.dataset)
    print(f"Dataset: {csv_path}")
    df = load_and_clean(csv_path)
    print(f"Loaded {len(df):,} rows, {df['video_id'].nunique()} videos, {df['player'].nunique()} players\n")

    sequences = build_sequences(df)
    slowmo_map = load_slowmo_map(df)

    if args.normalize_slowmo:
        n = normalize_slowmo_durations(sequences, slowmo_map, args.slowmo_target_duration)
        print(f"Resampled {n} slow-motion videos to ~{args.slowmo_target_duration}s each.\n")

    # -- split ---------------------------------------------------------------
    if args.split == "player_held_out":
        train_ids, test_ids = player_held_out_split(sequences, args.test_video_target, args.seed)
    elif args.split == "random_video":
        train_ids, test_ids = random_video_split(sequences, args.test_fraction, args.seed)
    elif args.split == "normal_speed_player_held_out":
        train_ids, test_ids = normal_speed_player_held_out_split(
            sequences, slowmo_map, args.test_video_target
        )
    else:
        raise ValueError(f"Unknown split strategy: {args.split}")

    summarize_split(sequences, train_ids, test_ids)

    # Optionally carve a validation slice out of TRAIN so checkpoint selection
    # never touches the test set (the original notebooks selected on test).
    eval_ids = test_ids
    if args.val_from_train > 0:
        rng = np.random.default_rng(args.seed)
        shuffled = list(rng.permutation(sorted(train_ids)))
        n_val = max(1, int(round(len(shuffled) * args.val_from_train)))
        eval_ids = shuffled[:n_val]
        train_ids = shuffled[n_val:]
        print(f"\nValidation carved from train: {len(eval_ids)} videos "
              f"(checkpoint selection uses this, not test)")
    print()

    model = build_model(device=device)
    print(f"Receptive field: {model.receptive_field()} frames "
          f"(~{model.receptive_field() / 60:.2f}s at 60fps)\n")

    config = TrainConfig(
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, smoothing=args.smoothing, seed=args.seed,
    )
    checkpoint_path = paths.checkpoint(args.output)
    model, history = train_model(
        model, sequences, train_ids, eval_ids, checkpoint_path, config, device
    )

    if args.plot:
        plot_history(history, Path(args.plot))

    # -- final report on the real test split, from the best checkpoint --------
    from .train.evaluate import evaluate_boundaries, per_phase_error_table, pooled_error_stats_ms

    best = load_checkpoint(checkpoint_path, device)
    train_bd = evaluate_boundaries(best, sequences, train_ids, device=device)
    test_bd = evaluate_boundaries(best, sequences, test_ids, device=device)
    summary = pd.DataFrame({
        "train": pooled_error_stats_ms(train_bd),
        "test": pooled_error_stats_ms(test_bd),
    }).round(2)
    print("\nOverall pooled boundary error (ms) -- all phases, start+end combined:")
    print(summary.to_string())
    print("\nPer-phase test boundary error (ms):")
    print(per_phase_error_table(test_bd).round(1).to_string(index=False))

    return 0


# ── evaluate ───────────────────────────────────────────────────────────────
def cmd_evaluate(args) -> int:
    from .generate_dataset.dataset import build_sequences, load_and_clean, load_slowmo_map
    from .train.evaluate import (
        classification_report_text,
        frame_accuracy,
        per_phase_error_table,
        per_player_error_table,
        plot_confusion_matrix,
        pooled_error_stats_ms,
        train_vs_eval_report,
    )
    from .generate_dataset.features import normalize_slowmo_durations
    from .train.model import load_checkpoint, resolve_device
    from .train.splits import (
        normal_speed_player_held_out_split,
        player_held_out_split,
        random_video_split,
        summarize_split,
    )

    paths = DataPaths(args.data_root)
    device = resolve_device(args.device)

    df = load_and_clean(paths.dataset(args.dataset))
    sequences = build_sequences(df)
    slowmo_map = load_slowmo_map(df)

    if args.normalize_slowmo:
        n = normalize_slowmo_durations(sequences, slowmo_map, args.slowmo_target_duration)
        print(f"Resampled {n} slow-motion videos to ~{args.slowmo_target_duration}s each.\n")

    if args.split == "player_held_out":
        train_ids, test_ids = player_held_out_split(sequences, args.test_video_target, args.seed)
    elif args.split == "random_video":
        train_ids, test_ids = random_video_split(sequences, args.test_fraction, args.seed)
    else:
        train_ids, test_ids = normal_speed_player_held_out_split(
            sequences, slowmo_map, args.test_video_target
        )
    summarize_split(sequences, train_ids, test_ids)
    print()

    model = load_checkpoint(paths.checkpoint(args.model), device)

    summary, train_bd, test_bd = train_vs_eval_report(model, sequences, train_ids, test_ids, device)
    print("Overall pooled boundary error (ms) -- all phases, start+end combined:")
    print(summary.round(2).to_string())

    gap = summary.loc["mean_ms", "eval"] - summary.loc["mean_ms", "train"]
    if gap > 0:
        print(f"\nTest mean error is {gap:.1f}ms higher than train -- generalization gap.")
    else:
        print("\nTest mean error is not higher than train.")

    print("\nPer-phase test boundary error (ms):")
    print(per_phase_error_table(test_bd).round(1).to_string(index=False))

    print("\nPer-player test boundary error (ms):")
    print(per_player_error_table(test_bd).round(1).to_string(index=False))

    raw_acc, decoded_acc, true, decoded = frame_accuracy(model, sequences, test_ids, device)
    print(f"\nRaw per-frame accuracy    : {raw_acc:.4f}")
    print(f"Decoded (monotonic) accuracy: {decoded_acc:.4f}")
    print("\nPer-class report (decoded):")
    print(classification_report_text(true, decoded))

    if args.confusion_matrix:
        plot_confusion_matrix(true, decoded, Path(args.confusion_matrix))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        test_bd.to_csv(out, index=False)
        print(f"Per-boundary errors -> {out}")

    return 0


# ── predict ────────────────────────────────────────────────────────────────
def cmd_predict(args) -> int:
    from .train.model import load_checkpoint, resolve_device, resolve_yolo_device
    from .pose_estimation.pose import excluded_stems, load_pose_model, resolve_video_paths
    from .predict.predict import already_done_ids, predict_batch

    paths = DataPaths(args.data_root)
    device = resolve_device(args.device)
    yolo_device = resolve_yolo_device(args.yolo_device)
    print(f"TCN device: {device}  |  YOLO device: {yolo_device}")

    model = load_checkpoint(paths.checkpoint(args.model), device)

    yolo_path = Path(args.yolo_model) if args.yolo_model else paths.yolo_pose_model
    pose_model = load_pose_model(yolo_path, yolo_device)

    video_inputs = args.videos or [str(paths.video_dir)]
    all_videos = resolve_video_paths(video_inputs)
    exclude = excluded_stems(args.exclude_dir, args.exclude_ids_csv, args.exclude_suffix)
    videos = [v for v in all_videos if v.stem not in exclude]
    print(f"Found {len(all_videos)} videos, {len(exclude)} excluded ids, {len(videos)} candidates")

    if args.num_shards > 1:
        videos = videos[args.shard_index::args.num_shards]
        print(f"Shard {args.shard_index}/{args.num_shards}: {len(videos)} videos")

    output_path = Path(args.output)
    if args.resume:
        done = already_done_ids(output_path)
        if done:
            videos = [v for v in videos if v.stem not in done]
            print(f"--resume: {len(done)} already done, {len(videos)} remaining")

    predict_batch(
        videos, model, pose_model, output_path, device,
        batch_size=args.batch_size, max_missed_frac=args.max_missed_frac,
        resume=args.resume, allow_no_phase_reset=args.allow_no_phase_reset,
    )
    return 0


# ── analyze-predictions ────────────────────────────────────────────────────
def cmd_analyze_predictions(args) -> int:
    from .review_pipeline.confidence import (
        add_composite_confidence,
        add_missing_phase_columns,
        get_real_missing_phase_videos,
    )
    from .constants import PHASE_COL_PREFIXES

    df = pd.read_csv(args.predictions)
    print(f"Total videos in {Path(args.predictions).name}: {len(df)}")

    add_missing_phase_columns(df)
    add_composite_confidence(df)

    missing_mask = pd.DataFrame({
        prefix: df[f"{prefix}_start_frame"].isna() for prefix in PHASE_COL_PREFIXES
    })
    real_missing = get_real_missing_phase_videos(df)

    print(f"Total missing-phase instances: {int(df['n_missing_phases'].sum())}")
    print(f"Videos with >=1 missing phase: {int((df['n_missing_phases'] > 0).sum())} / {len(df)}")
    print(f"Videos where only 'no_phase' is missing (benign): {int(df['only_no_phase_missing'].sum())}")

    print("\nMissing-phase counts by phase:")
    print(missing_mask.sum().sort_values(ascending=False).to_string())

    print(f"\nVideos with a real missing phase ({len(real_missing)}):")
    cols = ["video_id", "n_missing_phases", "missing_phases", "composite_confidence"]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(real_missing[cols].sort_values("n_missing_phases", ascending=False).to_string(index=False))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        real_missing[cols].sort_values("n_missing_phases", ascending=False).to_csv(out, index=False)
        print(f"\nSaved -> {out}")
    return 0


# ── build-review-set ───────────────────────────────────────────────────────
def cmd_build_review_set(args) -> int:
    import numpy as np

    from .review_pipeline.confidence import (
        BAND_SPECS,
        add_composite_confidence,
        add_missing_phase_columns,
        extract_player,
        get_real_missing_phase_videos,
        plot_confidence_distribution,
        stratified_sample_by_player,
    )

    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.predictions)
    df["player"] = df["video_id"].apply(extract_player)
    add_missing_phase_columns(df)
    add_composite_confidence(df)

    # Bucket 1: every video with a genuinely missing phase. These are the
    # strongest failure signal available, and the population is small enough
    # that sampling would only lose information.
    missing_df = get_real_missing_phase_videos(df).copy()
    missing_df["confidence_band"] = "missing_phase"
    missing_df["review_priority"] = 1

    # Buckets 2..n: stratified sampling within fixed confidence bands.
    pool_df = df[~df["video_id"].isin(missing_df["video_id"])].copy()

    band_specs = BAND_SPECS
    if args.band_sizes:
        if len(args.band_sizes) != len(BAND_SPECS):
            raise ValueError(f"--band-sizes needs {len(BAND_SPECS)} values, got {len(args.band_sizes)}")
        band_specs = [(n, lo, hi, size) for (n, lo, hi, _), size in zip(BAND_SPECS, args.band_sizes)]

    print("=" * 78)
    print("SAMPLING PLAN")
    print("=" * 78)
    print(f"missing_phase              -- taking all {len(missing_df)}")

    band_frames = []
    for priority, (name, lo, hi, n_target) in enumerate(band_specs, start=2):
        band_pool = pool_df[
            (pool_df["composite_confidence"] > lo) & (pool_df["composite_confidence"] <= hi)
        ]
        sampled, alloc, fallback = stratified_sample_by_player(band_pool, n_target, rng)
        sampled = sampled.copy()
        sampled["confidence_band"] = name
        sampled["review_priority"] = priority
        band_frames.append(sampled)

        print(f"\n{name} ({lo:.2f}, {hi:.2f}]  -- target {n_target}, pool {len(band_pool)}, sampled {len(sampled)}:")
        counts = band_pool["player"].value_counts()
        for player in counts.index:
            n_sampled = int(alloc.get(player, 0))
            if n_sampled:
                print(f"  {player:<15s} {n_sampled:>3d} / {counts[player]:>3d} available")
        if fallback:
            print("  Fallback to all-available (population too small to sample proportionally):")
            for player, n in fallback:
                print(f"    {player}: took all {n}")

    review_df = pd.concat([missing_df] + band_frames, ignore_index=True)
    tail_cols = ["min_confidence", "mean_confidence", "composite_confidence",
                 "confidence_band", "review_priority"]
    lead_cols = [c for c in df.columns if c not in tail_cols]
    review_df = review_df[lead_cols + tail_cols].sort_values(
        ["review_priority", "composite_confidence"], ascending=[True, True]
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(out, index=False)

    print("\n" + "=" * 78)
    print("FINAL REVIEW SET")
    print("=" * 78)
    print(review_df["confidence_band"].value_counts().to_string())
    print(f"\nSaved {len(review_df)} rows -> {out}")

    if args.plot:
        plot_confidence_distribution(pool_df, Path(args.plot), band_specs)
    return 0


# ── resolve-review ─────────────────────────────────────────────────────────
def cmd_resolve_review(args) -> int:
    from .generate_dataset.indexing import INDEX_COLUMNS
    from .review_pipeline.review import resolve_auto_accepted, resolve_reviewed

    paths = DataPaths(args.data_root)

    review = pd.read_csv(args.audit)
    preds_all = pd.read_csv(args.predictions)
    preds = preds_all.set_index("video_id")
    pool_meta = pd.read_csv(args.pool_metadata or paths.pool_metadata_csv).set_index("video_id")

    correction_dirs = [Path(d) for d in (args.corrections or [])]

    reviewed_rows, reviewed_skipped = resolve_reviewed(
        review, preds, pool_meta, correction_dirs, paths.root
    )

    pseudo_rows, pseudo_skipped = [], []
    if not args.no_auto_accept:
        pseudo_rows, pseudo_skipped = resolve_auto_accepted(
            preds_all, set(review["video_id"]), pool_meta, args.threshold
        )

    out_df = pd.DataFrame(reviewed_rows + pseudo_rows, columns=INDEX_COLUMNS)
    out = paths.dataset(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)

    print(f"Reviewed videos in {Path(args.audit).name}: {len(review)}")
    print(f"  resolved: {len(reviewed_rows)}  skipped: {len(reviewed_skipped)}")
    for video_id, band, verdict, reason in reviewed_skipped:
        print(f"    [SKIP] {str(video_id):25s} band={str(band):15s} verdict={str(verdict)!r:12s} -> {reason}")

    if not args.no_auto_accept:
        print(f"\nUnreviewed predictions (composite confidence > {args.threshold}):")
        print(f"  resolved: {len(pseudo_rows)}  skipped: {len(pseudo_skipped)}")

    print(f"\nTotal new videos this round: {len(out_df)}")
    if not out_df.empty:
        print(out_df["label_source"].value_counts().to_string())
    print(f"\nSaved -> {out}")
    return 0


# ── export ─────────────────────────────────────────────────────────────────
def cmd_export(args) -> int:
    from .train.export import export_coreml, export_torchscript
    from .train.model import load_checkpoint

    paths = DataPaths(args.data_root)
    model = load_checkpoint(paths.checkpoint(args.model), "cpu")

    if args.format in ("torchscript", "both"):
        out = args.torchscript_output or "PhaseTCN.pt"
        export_torchscript(model, out, args.trace_seq_len)

    if args.format in ("mlpackage", "both"):
        out = args.coreml_output or "PhaseTCN.mlpackage"
        export_coreml(model, out, args.trace_seq_len, args.max_seq_len, args.ios_target)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase-detect",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # build-index
    p = sub.add_parser("build-index", help="Index chaptered videos into a video index CSV")
    _add_common(p)
    p.add_argument("--videos", default=None, help="Directory of chaptered videos (default: <data-root>/ground_truth)")
    p.add_argument("--output", default="video_data.csv")
    p.add_argument("--include-slowmo", action="store_true",
                   help="Keep slow-motion videos (excluded by default, matching the original ground-truth build)")
    p.set_defaults(func=cmd_build_index)

    # pool-metadata
    p = sub.add_parser("pool-metadata", help="Probe the whole video pool for fps/is_slowmo")
    _add_common(p)
    p.add_argument("--videos", default=None, help="Directory to probe (default: <data-root>/all)")
    p.add_argument("--output", default=None, help="Default: <data-root>/video_pool_metadata.csv")
    p.set_defaults(func=cmd_pool_metadata)

    # extract-pose
    p = sub.add_parser("extract-pose", help="Run pose estimation + labelling over an indexed video set")
    _add_common(p)
    p.add_argument("--index", required=True, help="Video index CSV (from build-index or resolve-review)")
    p.add_argument("--output", required=True, help="Output per-frame dataset CSV")
    p.add_argument("--yolo-model", default=None)
    p.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"])
    p.add_argument("--limit", type=int, default=None, help="Only process the first N videos (smoke test)")
    p.set_defaults(func=cmd_extract_pose)

    # merge-datasets
    p = sub.add_parser("merge-datasets", help="Concatenate a new round onto the combined training set")
    _add_common(p)
    p.add_argument("--previous", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source-tag", default="new", help="Value written to the dataset_source column")
    p.set_defaults(func=cmd_merge_datasets)

    # train
    p = sub.add_parser("train", help="Train a PhaseTCN")
    _add_common(p)
    p.add_argument("--dataset", required=True, help="Joined pose+label CSV")
    p.add_argument("--output", required=True, help="Checkpoint path (.pt)")
    p.add_argument("--split", default="player_held_out",
                   choices=["player_held_out", "random_video", "normal_speed_player_held_out"])
    p.add_argument("--test-video-target", type=int, default=16,
                   help="Approx. videos to hold out (player-held-out splits)")
    p.add_argument("--test-fraction", type=float, default=0.2, help="random_video split only")
    p.add_argument("--val-from-train", type=float, default=0.0,
                   help="Fraction of TRAIN to hold out for checkpoint selection, so it never touches test")
    p.add_argument("--normalize-slowmo", action="store_true",
                   help="Time-compress slow-motion clips to the normal-speed mean duration")
    p.add_argument("--slowmo-target-duration", type=float, default=2.4)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--smoothing", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"])
    p.add_argument("--plot", default=None, help="Path to save training curves PNG")
    p.set_defaults(func=cmd_train)

    # evaluate
    p = sub.add_parser("evaluate", help="Boundary-error and accuracy report for a checkpoint")
    _add_common(p)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="player_held_out",
                   choices=["player_held_out", "random_video", "normal_speed_player_held_out"])
    p.add_argument("--test-video-target", type=int, default=16)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--normalize-slowmo", action="store_true")
    p.add_argument("--slowmo-target-duration", type=float, default=2.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"])
    p.add_argument("--output", default=None, help="Write per-boundary errors to this CSV")
    p.add_argument("--confusion-matrix", default=None, help="Path to save a confusion-matrix PNG")
    p.set_defaults(func=cmd_evaluate)

    # predict
    p = sub.add_parser("predict", help="Run inference over unlabeled videos")
    _add_common(p)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--videos", nargs="+", default=None,
                   help="Directories, video files, or .txt lists (default: <data-root>/all)")
    p.add_argument("--exclude-dir", nargs="*", default=[])
    p.add_argument("--exclude-ids-csv", nargs="*", default=[])
    p.add_argument("--exclude-suffix", default=" CHAPTERED")
    p.add_argument("--yolo-model", default=None)
    p.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"])
    p.add_argument("--yolo-device", default=None, choices=["cpu", "mps", "cuda"])
    p.add_argument("--batch-size", type=int, default=16, help="Frames per YOLO call")
    p.add_argument("--max-missed-frac", type=float, default=0.3)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--allow-no-phase-reset", action="store_true",
                   help="Use the training-style decode. Off by default so every phase gets a "
                        "contiguous span and a missing phase stays a meaningful signal.")
    p.set_defaults(func=cmd_predict)

    # analyze-predictions
    p = sub.add_parser("analyze-predictions", help="Missing-phase report over a predictions CSV")
    _add_common(p)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_analyze_predictions)

    # build-review-set
    p = sub.add_parser("build-review-set", help="Stratified human-review set from predictions")
    _add_common(p)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--band-sizes", type=int, nargs="+", default=None,
                   help="Sample size per confidence band (very_low low_mid mid mid_high high)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", default=None, help="Path to save the confidence-distribution PNG")
    p.set_defaults(func=cmd_build_review_set)

    # resolve-review
    p = sub.add_parser("resolve-review", help="Turn audit results into the next round's video index")
    _add_common(p)
    p.add_argument("--audit", required=True, help="Audit results CSV (video_id, band, verdict, ...)")
    p.add_argument("--predictions", required=True, help="Predictions CSV the audit was performed against")
    p.add_argument("--output", required=True, help="Output video index CSV")
    p.add_argument("--corrections", nargs="*", default=[],
                   help="Directories containing reviewer-corrected {video_id}_Labelled.mp4 exports")
    p.add_argument("--pool-metadata", default=None)
    p.add_argument("--threshold", type=float, default=0.55,
                   help="Composite-confidence threshold for auto-accepting unreviewed predictions")
    p.add_argument("--no-auto-accept", action="store_true",
                   help="Only resolve audited videos; skip confidence-based auto-acceptance")
    p.set_defaults(func=cmd_resolve_review)

    # export
    p = sub.add_parser("export", help="Export a checkpoint to TorchScript and/or Core ML")
    _add_common(p)
    p.add_argument("--model", required=True)
    p.add_argument("--format", default="both", choices=["torchscript", "mlpackage", "both"])
    p.add_argument("--torchscript-output", default=None)
    p.add_argument("--coreml-output", default=None)
    p.add_argument("--trace-seq-len", type=int, default=90)
    p.add_argument("--max-seq-len", type=int, default=600)
    p.add_argument("--ios-target", default="iOS16")
    p.set_defaults(func=cmd_export)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
