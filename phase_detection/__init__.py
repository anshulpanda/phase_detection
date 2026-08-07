"""
phase_detection -- tennis forehand swing-phase detection.

A pose-based Temporal Convolutional Network that labels every frame of a
forehand clip with its swing phase, plus the active-learning pipeline used to
grow its training set from 82 hand-labelled videos to 884.

The data (videos, datasets, checkpoints) lives outside this repository; see
`paths.DataPaths` and README.md for how it is located.
"""

from .constants import (
    ID_TO_PHASE,
    NUM_CLASSES,
    PHASE_ORDER,
    PHASE_TO_ID,
    POSE_FEATURE_COLS,
)
from .decode import get_phase_boundaries, monotonic_decode
from .train.model import PhaseTCN, build_model, load_checkpoint, save_checkpoint
from .paths import DataPaths

__version__ = "0.1.0"

__all__ = [
    "PHASE_ORDER",
    "PHASE_TO_ID",
    "ID_TO_PHASE",
    "NUM_CLASSES",
    "POSE_FEATURE_COLS",
    "PhaseTCN",
    "build_model",
    "load_checkpoint",
    "save_checkpoint",
    "monotonic_decode",
    "get_phase_boundaries",
    "DataPaths",
    "__version__",
]
