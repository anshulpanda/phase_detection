"""
Feature normalization and time resampling.

Everything here operates on raw pixel-space keypoints and must stay bit-identical
between training and inference -- a mismatch silently degrades predictions
rather than raising, so both paths call these same functions.
"""

import numpy as np
import pandas as pd

from ..constants import POSE_FEATURE_COLS


def normalize_pose(frame_df: pd.DataFrame) -> np.ndarray:
    """Make keypoints invariant to where the player is in frame and how big.

    Centers on the hip midpoint and scales by torso length (shoulder midpoint
    to hip midpoint). Confidence columns pass through untouched -- they are
    already 0-1 and carry no spatial meaning.

    Returns (n_frames, 51) float32.
    """
    out = frame_df[POSE_FEATURE_COLS].to_numpy(dtype=np.float32).copy()

    hip_mid_x = (frame_df["left_hip_x"].to_numpy() + frame_df["right_hip_x"].to_numpy()) / 2.0
    hip_mid_y = (frame_df["left_hip_y"].to_numpy() + frame_df["right_hip_y"].to_numpy()) / 2.0
    shoulder_mid_x = (frame_df["left_shoulder_x"].to_numpy() + frame_df["right_shoulder_x"].to_numpy()) / 2.0
    shoulder_mid_y = (frame_df["left_shoulder_y"].to_numpy() + frame_df["right_shoulder_y"].to_numpy()) / 2.0

    torso_len = np.sqrt((shoulder_mid_x - hip_mid_x) ** 2 + (shoulder_mid_y - hip_mid_y) ** 2)
    # Guard a degenerate torso (both midpoints coincident) rather than dividing by ~0.
    torso_len = np.where(torso_len < 1e-6, 1.0, torso_len)

    for i, col in enumerate(POSE_FEATURE_COLS):
        if col.endswith("_x"):
            out[:, i] = (out[:, i] - hip_mid_x) / torso_len
        elif col.endswith("_y"):
            out[:, i] = (out[:, i] - hip_mid_y) / torso_len
        # _conf columns left as-is

    return out


def handle_missed_detections(vdf: pd.DataFrame, max_missed_frac: float = 0.3) -> pd.DataFrame:
    """Interpolate frames where the pose model found no person.

    Short gaps (a brief occlusion) are linearly interpolated and then
    edge-filled. But a video where a large fraction of frames had no detection
    is a fundamentally bad capture, not noise -- those raise instead, so the
    caller can flag the video for review rather than trusting a prediction
    built mostly from interpolation.
    """
    missed_frac = vdf[POSE_FEATURE_COLS[0]].isna().mean()
    if missed_frac > max_missed_frac:
        raise ValueError(
            f"{missed_frac * 100:.1f}% of frames had no person detected -- "
            f"exceeds the {max_missed_frac * 100:.0f}% threshold. "
            f"Flag this video for manual review instead of trusting predictions on it."
        )
    vdf = vdf.copy()
    vdf[POSE_FEATURE_COLS] = vdf[POSE_FEATURE_COLS].interpolate(limit_direction="both")
    return vdf


def resample_sequence(features: np.ndarray, labels: np.ndarray, target_n_frames: int) -> tuple:
    """Resample a (T, F) feature sequence and (T,) label sequence in time.

    Continuous pose trajectories are linearly interpolated, preserving the
    shape of the motion. Phase labels use nearest-neighbour instead -- an
    ordinal phase id cannot be meaningfully interpolated.
    """
    n_frames = features.shape[0]
    target_n_frames = max(2, target_n_frames)
    if n_frames == target_n_frames:
        return features, labels

    orig_t = np.linspace(0.0, 1.0, n_frames)
    target_t = np.linspace(0.0, 1.0, target_n_frames)

    resampled_features = np.empty((target_n_frames, features.shape[1]), dtype=np.float32)
    for i in range(features.shape[1]):
        resampled_features[:, i] = np.interp(target_t, orig_t, features[:, i])

    nearest_idx = np.clip(np.round(target_t * (n_frames - 1)).astype(int), 0, n_frames - 1)
    resampled_labels = labels[nearest_idx]

    return resampled_features, resampled_labels


def normalize_slowmo_durations(
    sequences: dict,
    slowmo_map: pd.Series,
    target_duration_s: float = 2.4,
) -> int:
    """Time-compress slow-motion clips to the normal-speed mean duration.

    Slow-motion clips average ~10.4s against ~2.4s for normal speed, far beyond
    the model's ~2s receptive field -- which was root-caused as the reason a
    handful of held-out players had catastrophically bad boundary predictions.
    Resampling preserves the whole swing rather than discarding the clip.

    Each video is scaled relative to its *own* original duration (fps is left
    untouched), so a 10s and a 20s clip both land at ~target_duration_s at
    their own appropriate compression ratio.

    Mutates `sequences` in place; returns how many videos were resampled.
    """
    n_resampled = 0
    for video_id, item in sequences.items():
        if not slowmo_map.get(video_id, False):
            continue
        current_n_frames = len(item["labels"])
        current_duration = current_n_frames / item["fps"]
        target_n_frames = round(current_n_frames * (target_duration_s / current_duration))
        item["features"], item["labels"] = resample_sequence(
            item["features"], item["labels"], target_n_frames
        )
        n_resampled += 1
    return n_resampled
