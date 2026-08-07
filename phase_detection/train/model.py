"""
The PhaseTCN architecture.

A small Temporal Convolutional Network over per-frame pose keypoints,
classifying each frame into an ordinal swing phase.

Why a TCN: it is cheap and parallelizable (a good fit for eventual on-device
deployment), it has a fixed and inspectable receptive field that can be sized
against the known ~2s maximum swing duration, and it trains stably on a small
dataset.

The network is deliberately non-causal ('same' padding, both-directions
context). At inference the model runs over a whole bounded clip rather than a
live frame-by-frame stream, so full context is available -- and the swing
boundaries are the *output*, not a precondition.
"""

from pathlib import Path

import torch
import torch.nn as nn

from ..constants import NUM_CLASSES, N_POSE_FEATURES

DEFAULT_CHANNELS = (64, 64, 64, 64)
DEFAULT_KERNEL_SIZE = 5
DEFAULT_DROPOUT = 0.2


class TemporalBlock(nn.Module):
    """conv -> norm -> activation -> dropout, twice, with a residual connection."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        # 'same' padding -- non-causal, we have full context of the clip.
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.act(self.norm2(self.conv2(out)))
        out = self.dropout(out)
        return self.act(out + residual)


class PhaseTCN(nn.Module):
    """Stacked dilated temporal blocks over (batch, time, features) input."""

    def __init__(
        self,
        n_features: int = N_POSE_FEATURES,
        num_classes: int = NUM_CLASSES,
        channels: tuple = DEFAULT_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.n_features = n_features
        self.num_classes = num_classes
        self.channels = tuple(channels)
        self.kernel_size = kernel_size

        self.input_proj = nn.Conv1d(n_features, channels[0], kernel_size=1)

        blocks = []
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i  # 1, 2, 4, 8 -- grows the receptive field cheaply
            blocks.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv1d(in_ch, num_classes, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, time, features) -> (batch, features, time)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        logits = self.head(x)
        return logits.transpose(1, 2)  # -> (batch, time, num_classes)

    def receptive_field(self) -> int:
        """Frames of context each output frame can see.

        Two conv layers per block, each contributing (kernel_size - 1) * dilation
        on both sides.
        """
        rf = 1
        for i in range(len(self.channels)):
            rf += 2 * (self.kernel_size - 1) * (2 ** i)
        return rf


class PhaseTCNWithSoftmax(nn.Module):
    """Wraps PhaseTCN so an exported graph emits probabilities, not logits."""

    def __init__(self, model: PhaseTCN):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.softmax(self.model(x), dim=-1)


def build_model(
    n_features: int = N_POSE_FEATURES,
    channels: tuple = DEFAULT_CHANNELS,
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    dropout: float = DEFAULT_DROPOUT,
    device: torch.device | str = "cpu",
) -> PhaseTCN:
    model = PhaseTCN(
        n_features=n_features,
        num_classes=NUM_CLASSES,
        channels=tuple(channels),
        kernel_size=kernel_size,
        dropout=dropout,
    )
    return model.to(device)


def save_checkpoint(model: PhaseTCN, path: str | Path) -> None:
    """Save weights plus the architecture needed to rebuild them.

    Older checkpoints in this project were saved as a bare state_dict; this
    richer format is what `load_checkpoint` prefers, and it stays
    backward-compatible in both directions.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_features": model.n_features,
            "channels": list(model.channels),
            "kernel_size": model.kernel_size,
            "num_classes": model.num_classes,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
    n_features: int = N_POSE_FEATURES,
    channels: tuple = DEFAULT_CHANNELS,
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    dropout: float = DEFAULT_DROPOUT,
) -> PhaseTCN:
    """Load a checkpoint saved either as a bare state_dict (the original
    convention, used by every checkpoint trained from the notebooks) or as the
    richer dict written by `save_checkpoint`. When architecture metadata is
    present in the file it wins over the arguments passed here.
    """
    obj = torch.load(path, map_location=device)

    if isinstance(obj, dict) and "state_dict" in obj:
        n_features = obj.get("n_features", n_features)
        channels = tuple(obj.get("channels", channels))
        kernel_size = obj.get("kernel_size", kernel_size)
        dropout = obj.get("dropout", dropout)
        state_dict = obj["state_dict"]
    else:
        state_dict = obj

    model = build_model(n_features, channels, kernel_size, dropout, device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def resolve_device(preference: str | None = None) -> torch.device:
    """Pick a torch device, honouring an explicit preference when given."""
    if preference:
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_yolo_device(preference: str | None = None) -> str:
    """YOLO benefits from MPS on Apple silicon, where the TCN does not."""
    if preference:
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
