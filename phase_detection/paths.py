"""
Data-root resolution.

No dataset, video, or model weight lives inside this repo -- the code is the
deliverable, the data stays wherever the user keeps it (historically
`stealthy_wealthy/phase_model/`). Every path is resolved off a single
configurable root so the pipeline can be pointed at a different copy of the
data without editing code.

Resolution order (first hit wins):
  1. an explicit `--data-root` passed on the command line
  2. the PHASE_DATA_ROOT environment variable (a .env file next to the repo
     root is read if present)
  3. ../phase_model relative to this repository
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "phase_model"

_ENV_VAR = "PHASE_DATA_ROOT"


def _load_dotenv() -> None:
    """Minimal .env reader -- avoids a python-dotenv dependency for one variable.

    Only sets variables that are not already present in the real environment,
    so an explicit `export` always beats the file.
    """
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def resolve_data_root(override: str | Path | None = None) -> Path:
    """Resolve the data root, raising if it does not exist."""
    if override is not None:
        root = Path(override)
    else:
        _load_dotenv()
        env_value = os.environ.get(_ENV_VAR)
        root = Path(env_value) if env_value else DEFAULT_DATA_ROOT

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Data root does not exist: {root}\n"
            f"Set it with --data-root, or the {_ENV_VAR} environment variable, "
            f"or a .env file at {REPO_ROOT / '.env'} (see .env.example)."
        )
    return root


class DataPaths:
    """Well-known locations inside the data root.

    Only the directory layout is fixed here; individual dataset filenames are
    passed explicitly by the CLI so that per-round artifacts
    (joined_dataset_round2.csv, ...) stay visible in the command that
    produced them rather than being hidden behind a constant.
    """

    def __init__(self, root: str | Path | None = None):
        self.root = resolve_data_root(root)

    # -- source video pools -------------------------------------------------
    @property
    def video_dir(self) -> Path:
        """The full unlabeled video pool (`all/`)."""
        return self.root / "all"

    @property
    def ground_truth_dir(self) -> Path:
        """Hand-chaptered ground-truth videos."""
        return self.root / "ground_truth"

    @property
    def pool_metadata_csv(self) -> Path:
        """Per-video fps/frame-count/is_slowmo index for the whole pool."""
        return self.root / "video_pool_metadata.csv"

    # -- models -------------------------------------------------------------
    @property
    def yolo_pose_model(self) -> Path:
        return self.root / "pose_estimation" / "yolo26n-pose.pt"

    # -- generated artifacts ------------------------------------------------
    @property
    def datasets_dir(self) -> Path:
        """Where dataset CSVs are read from and written to."""
        return self.root / "training_pipeline" / "generate_dataset"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "training_pipeline" / "train"

    def dataset(self, name: str) -> Path:
        """Resolve a dataset CSV: absolute paths pass through, bare names
        resolve inside `datasets_dir`."""
        p = Path(name)
        return p if p.is_absolute() else self.datasets_dir / p

    def checkpoint(self, name: str) -> Path:
        p = Path(name)
        return p if p.is_absolute() else self.checkpoints_dir / p

    def __repr__(self) -> str:
        return f"DataPaths(root={self.root})"
