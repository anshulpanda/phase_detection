"""
Pose + label extraction: video index -> per-frame training dataset.

For every frame of every indexed video, runs the pose model and assigns the
phase label whose chapter contains that frame's timestamp. Frames with no
detected person are dropped entirely rather than imputed -- this is the
training path, where a fabricated skeleton would be a fabricated label. (The
*inference* path interpolates short gaps instead, since there it must produce
an answer for a given video; see features.handle_missed_detections.)
"""

import ast
from pathlib import Path

import cv2
import pandas as pd

from ..constants import (
    FRAME_META_COLS,
    POSE_FEATURE_COLS,
    PHASE_TO_ID,
    canonical_label,
)
from ..pose_estimation.pose import extract_keypoints, label_for_frame


def extract_dataset(
    video_index: pd.DataFrame,
    data_root: Path,
    pose_model,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the per-frame pose+label dataset for an indexed set of videos."""
    all_rows = []
    n_missing_person = 0
    n_skipped_videos = 0

    for i, meta in video_index.iterrows():
        video_id = meta["video_id"]
        video_path = Path(meta["video_path"])
        if not video_path.is_absolute():
            video_path = data_root / video_path

        chapters = meta["chapters"]
        if isinstance(chapters, str):
            chapters = ast.literal_eval(chapters)

        if not video_path.exists():
            print(f"[SKIP] not found: {video_path}")
            n_skipped_videos += 1
            continue

        cap = cv2.VideoCapture(str(video_path))
        # The container's own fps is authoritative over the index's copy --
        # a corrected re-export can differ from what was recorded upstream.
        fps = cap.get(cv2.CAP_PROP_FPS) or float(meta["video_fps"])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if verbose:
            print(
                f"\n[{i + 1}/{len(video_index)}] {video_id}  "
                f"({total} frames @ {fps:.1f} fps  {frame_w}x{frame_h})  "
                f"[{meta.get('label_source', 'unknown')}]"
            )

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            label, ch = label_for_frame(frame_idx, fps, chapters)
            label = canonical_label(label)

            result = pose_model(frame, verbose=False)[0]
            kpts = extract_keypoints(result)
            if kpts is None:
                n_missing_person += 1
                frame_idx += 1
                continue

            row = {
                "video_id": video_id,
                "video_path": str(meta["video_path"]),
                "player": meta.get("player"),
                "frame_idx": frame_idx,
                "timestamp_s": round(frame_idx / fps, 6),
                "fps": fps,
                "frame_w": frame_w,
                "frame_h": frame_h,
                "video_total_frames": total,
                "video_duration_s": float(meta["video_duration_s"]),
                "is_slowmo": bool(meta["is_slowmo"]),
                "phase_label": label,
                "phase_label_id": PHASE_TO_ID.get(label, -1),
                "chapter_id": ch["chapter_id"],
                "chapter_start_s": ch["start_s"],
                "chapter_end_s": ch["end_s"],
                "chapter_duration_s": round(ch["end_s"] - ch["start_s"], 6),
                "time_since_chapter_start_s": round(frame_idx / fps - ch["start_s"], 6),
            }
            for col, val in zip(POSE_FEATURE_COLS, kpts.flatten()):
                row[col] = float(val)

            all_rows.append(row)
            frame_idx += 1

            if verbose and frame_idx % 100 == 0:
                print(f"  {frame_idx}/{total}")

        cap.release()
        if verbose:
            print(f"  done - {frame_idx} frames read")

    dataset = pd.DataFrame(all_rows)
    if dataset.empty:
        raise RuntimeError("No frames extracted -- check video paths in the index.")
    dataset = dataset[FRAME_META_COLS + POSE_FEATURE_COLS]

    if verbose:
        print(f"\nDropped {n_missing_person} frames with no detected person")
        if n_skipped_videos:
            print(f"Skipped {n_skipped_videos} videos whose file could not be found")

    return dataset


def merge_datasets(
    previous_csv: Path,
    new_csv: Path,
    output_csv: Path,
    source_tag: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """Concatenate a round's new data onto the running combined dataset.

    Guards against the two failure modes that silently corrupt a training set:
    a column-shape mismatch between rounds, and the same video appearing twice
    (which would leak a video across a train/test boundary later).
    """
    prev = pd.read_csv(previous_csv)
    new = pd.read_csv(new_csv)

    prev_cols = [c for c in prev.columns if c != "dataset_source"]
    if list(new.columns) != prev_cols:
        raise ValueError(
            "Column mismatch between the previous combined set and the new data:\n"
            f"  previous only: {sorted(set(prev_cols) - set(new.columns))}\n"
            f"  new only:      {sorted(set(new.columns) - set(prev_cols))}"
        )

    overlap = set(prev["video_id"]) & set(new["video_id"])
    if overlap:
        raise ValueError(f"video_id collision between rounds: {sorted(overlap)[:10]}")

    new = new.copy()
    new["dataset_source"] = source_tag
    if "dataset_source" not in prev.columns:
        prev = prev.copy()
        prev["dataset_source"] = "previous"

    combined = pd.concat([prev, new], ignore_index=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)

    if verbose:
        print(f"Previous : {prev['video_id'].nunique():4d} videos, {len(prev):8,} rows")
        print(f"New      : {new['video_id'].nunique():4d} videos, {len(new):8,} rows")
        print(f"Combined : {combined['video_id'].nunique():4d} videos, {len(combined):8,} rows")
        print("\nLabel distribution (combined):")
        print(combined["phase_label"].value_counts().to_string())
        print("\ndataset_source breakdown:")
        print(combined["dataset_source"].value_counts().to_string())
        print(f"\nSaved -> {output_csv}")

    return combined
