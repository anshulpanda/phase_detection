"""
Dataset loading and sequence assembly.

One `video_id` is one training sequence -- clips are already short, bounded
swings, so there is no windowing step.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..constants import PHASE_ORDER, PHASE_TO_ID, POSE_FEATURE_COLS
from .features import normalize_pose


def load_and_clean(csv_path: str | Path) -> pd.DataFrame:
    """Load a joined pose+label CSV and attach a clean ordinal `phase_id`.

    The CSV's own `phase_label_id` column is deliberately ignored: it is
    inconsistent across rows (`Acceleration` appears with ids 0, 1 *and* 2 on
    different rows, an artifact of per-video chapter numbering). The mapping is
    rebuilt from the `phase_label` text against the global PHASE_ORDER instead.

    Raises rather than silently coercing if labels or pose columns are missing,
    since either means the upstream extraction stage produced something the
    model cannot be trained on.
    """
    df = pd.read_csv(csv_path)

    present = set(df["phase_label"].unique())
    missing = [c for c in PHASE_ORDER if c not in present]
    extra = [l for l in present if l not in PHASE_TO_ID]
    if missing:
        raise ValueError(f"Expected phase labels not found in data: {missing}")
    if extra:
        raise ValueError(f"Unexpected phase labels in data not in PHASE_ORDER: {extra}")

    missing_pose = [c for c in POSE_FEATURE_COLS if c not in df.columns]
    if missing_pose:
        raise ValueError(f"Missing expected pose columns: {missing_pose}")

    df = df.copy()
    df["phase_id"] = df["phase_label"].map(PHASE_TO_ID)
    return df


def build_sequences(df: pd.DataFrame) -> dict:
    """Group a frame-level dataframe into per-video sequences.

    Returns {video_id: {"features": (T, 51), "labels": (T,), "player": str, "fps": float}}
    with pose features already normalized.
    """
    sequences = {}
    for video_id, vdf in df.groupby("video_id"):
        vdf = vdf.sort_values("frame_idx")
        sequences[video_id] = {
            "features": normalize_pose(vdf),
            "labels": vdf["phase_id"].to_numpy(dtype=np.int64),
            "player": vdf["player"].iloc[0],
            "fps": float(vdf["fps"].iloc[0]),
        }
    return sequences


def load_slowmo_map(df: pd.DataFrame) -> pd.Series:
    """video_id -> is_slowmo, taken from the dataset's own carried-through column.

    `is_slowmo` is detected once at ingest (a container format tag) and carried
    through every stage, so it can be read straight off the joined dataset
    rather than re-derived.
    """
    if "is_slowmo" not in df.columns:
        raise ValueError("Dataset has no `is_slowmo` column.")
    return df.drop_duplicates("video_id").set_index("video_id")["is_slowmo"]


class SwingSequenceDataset(Dataset):
    """Torch dataset over pre-built variable-length sequences."""

    def __init__(self, sequences: dict, video_ids: list):
        self.items = [sequences[v] for v in video_ids]
        self.video_ids = list(video_ids)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        return {
            "features": torch.from_numpy(item["features"].copy()).float(),
            "labels": torch.from_numpy(item["labels"].copy()).long(),
            "video_id": self.video_ids[idx],
        }


def collate_pad(batch: list) -> dict:
    """Pad a batch to its longest sequence and emit a validity mask.

    Padded label positions are set to -100 so they are trivially identifiable;
    the loss additionally masks them out explicitly.
    """
    max_len = max(item["features"].shape[0] for item in batch)
    n_features = batch[0]["features"].shape[1]

    padded_features = torch.zeros(len(batch), max_len, n_features)
    padded_labels = torch.full((len(batch), max_len), fill_value=-100, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    video_ids = []

    for i, item in enumerate(batch):
        seq_len = item["features"].shape[0]
        padded_features[i, :seq_len] = item["features"]
        padded_labels[i, :seq_len] = item["labels"]
        mask[i, :seq_len] = True
        video_ids.append(item["video_id"])

    return {
        "features": padded_features,
        "labels": padded_labels,
        "mask": mask,
        "video_ids": video_ids,
    }
