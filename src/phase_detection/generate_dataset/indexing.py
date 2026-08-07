"""
Video indexing: turn chaptered videos into a video-level index CSV.

The index is the contract between labelling and training. One row per video,
carrying fps/frame count/duration, the `is_slowmo` flag, and the chapter list
that defines phase boundaries. Every downstream stage reads this shape,
whether the chapters came from a hand-chaptered ground-truth video or from a
resolved review round.
"""

from pathlib import Path

import pandas as pd

from ..review_pipeline.confidence import extract_player
from ..constants import canonical_label
from ..pose_estimation.pose import get_video_metadata
from ..constants import VIDEO_EXTENSIONS

INDEX_COLUMNS = [
    "video_id", "video_path", "video_fps", "video_total_frames", "video_duration_s",
    "is_slowmo", "chapters", "num_chapters", "expected_frames",
    "player", "label_source", "band", "verdict",
]


def build_ground_truth_index(
    video_dir: Path,
    data_root: Path,
    exclude_slowmo: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Index every chaptered video in a directory via ffprobe.

    `exclude_slowmo` mirrors the original ground-truth build, which trained
    only on normal-speed footage. Slow-motion handling arrived later as an
    explicit time-normalization step (see features.normalize_slowmo_durations),
    so keep this True to reproduce the original ground-truth CSV.
    """
    videos = sorted(f for f in Path(video_dir).rglob("*") if f.suffix in VIDEO_EXTENSIONS)
    if verbose:
        print(f"Found {len(videos)} videos in {video_dir}")

    rows = []
    n_skipped_slowmo = 0
    n_skipped_nochapters = 0

    for i, video_path in enumerate(videos):
        meta = get_video_metadata(video_path)

        if exclude_slowmo and meta["is_slowmo"]:
            n_skipped_slowmo += 1
            continue
        if not meta["chapters"]:
            n_skipped_nochapters += 1
            if verbose:
                print(f"  [SKIP] no chapters: {video_path.name}")
            continue

        chapters = [
            {**ch, "label": canonical_label(ch["label"])} for ch in meta["chapters"]
        ]

        try:
            rel_path = str(video_path.relative_to(data_root))
        except ValueError:
            rel_path = str(video_path)

        video_id = video_path.stem
        rows.append({
            "video_id": video_id,
            "video_path": rel_path,
            "video_fps": meta["fps"],
            "video_total_frames": meta["total_frames"],
            "video_duration_s": meta["duration_s"],
            "is_slowmo": meta["is_slowmo"],
            "chapters": chapters,
            "num_chapters": len(chapters),
            "expected_frames": meta["duration_s"] * meta["fps"],
            "player": extract_player(video_id),
            "label_source": "ground_truth",
            "band": None,
            "verdict": None,
        })

        if verbose and (i + 1) % 25 == 0:
            print(f"  [{i + 1}/{len(videos)}] indexed")

    df = pd.DataFrame(rows, columns=INDEX_COLUMNS)

    if verbose:
        print(f"\nIndexed {len(df)} videos")
        if n_skipped_slowmo:
            print(f"  skipped {n_skipped_slowmo} slow-motion videos (--include-slowmo to keep)")
        if n_skipped_nochapters:
            print(f"  skipped {n_skipped_nochapters} videos with no chapter marks")

    return df


def build_pool_metadata(video_dir: Path, data_root: Path, verbose: bool = True) -> pd.DataFrame:
    """Lightweight per-video metadata over the whole pool (no chapters needed).

    This is the `is_slowmo`/fps lookup that the review-resolution stage joins
    against, so it must cover every video the model might predict on -- not
    just the labelled ones.
    """
    videos = sorted(f for f in Path(video_dir).rglob("*") if f.suffix in VIDEO_EXTENSIONS)
    if verbose:
        print(f"Probing {len(videos)} videos in {video_dir}")

    rows = []
    for i, video_path in enumerate(videos):
        meta = get_video_metadata(video_path)
        try:
            rel_path = str(video_path.relative_to(data_root))
        except ValueError:
            rel_path = str(video_path)
        rows.append({
            "video_id": video_path.stem,
            "video_path": rel_path,
            "fps": meta["fps"],
            "total_frames": meta["total_frames"],
            "duration_s": meta["duration_s"],
            "is_slowmo": meta["is_slowmo"],
        })
        if verbose and (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(videos)}] probed")

    return pd.DataFrame(rows)
