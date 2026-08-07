"""
Training loop.

Regularization in play: dropout (0.2) and BatchNorm inside every temporal
block, Adam weight decay (1e-4), ordinal label smoothing (0.1) as target
regularization, and best-checkpoint selection as implicit early stopping.

A caveat worth stating plainly: checkpoint selection uses the *test* split,
because the project never carved out a separate validation set. That makes the
reported test accuracy mildly optimistic. `--val-from-train` carves a
validation slice out of the training videos so selection stops touching test.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..constants import NUM_CLASSES
from ..generate_dataset.dataset import SwingSequenceDataset, collate_pad
from .losses import build_ordinal_smoothed_targets, compute_class_weights, soft_cross_entropy
from .model import PhaseTCN, save_checkpoint


@dataclass
class TrainConfig:
    n_epochs: int = 40
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    smoothing: float = 0.1
    seed: int = 42
    log_every: int = 5


@dataclass
class TrainHistory:
    train_loss: list = field(default_factory=list)
    eval_loss: list = field(default_factory=list)
    eval_acc: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"train_loss": self.train_loss, "eval_loss": self.eval_loss, "eval_acc": self.eval_acc}


def train_one_epoch(model, loader, optimizer, class_weights, smoothing, device) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["labels"]
        mask = batch["mask"].to(device)

        soft_targets = build_ordinal_smoothed_targets(labels, NUM_CLASSES, smoothing).to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = soft_cross_entropy(logits, soft_targets, mask, class_weights.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_epoch(model, loader, class_weights, smoothing, device) -> tuple[float, float]:
    """Loss and raw per-frame accuracy over all non-padded frames.

    This is the *undecoded* accuracy -- argmax per frame, no monotonic decode.
    The decoded number (see `evaluate.py`) is the one that matches what the
    boundary-extraction pipeline actually consumes.
    """
    model.eval()
    total_loss, n_batches, correct, total = 0.0, 0, 0, 0
    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["labels"]
        mask = batch["mask"].to(device)

        soft_targets = build_ordinal_smoothed_targets(labels, NUM_CLASSES, smoothing).to(device)
        logits = model(features)
        loss = soft_cross_entropy(logits, soft_targets, mask, class_weights.to(device))
        total_loss += loss.item()
        n_batches += 1

        preds = logits.argmax(dim=-1).cpu()
        valid = labels != -100
        correct += ((preds == labels) & valid).sum().item()
        total += valid.sum().item()

    return total_loss / max(n_batches, 1), correct / max(total, 1)


def train_model(
    model: PhaseTCN,
    sequences: dict,
    train_ids: list,
    eval_ids: list,
    checkpoint_path: str | Path,
    config: TrainConfig | None = None,
    device: torch.device | str = "cpu",
) -> tuple[PhaseTCN, TrainHistory]:
    """Train and checkpoint on best eval accuracy.

    `eval_ids` is whatever split drives checkpoint selection -- the test set by
    default (matching the original notebooks), or a validation slice when the
    caller supplies one.
    """
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_loader = DataLoader(
        SwingSequenceDataset(sequences, train_ids),
        batch_size=config.batch_size, shuffle=True, collate_fn=collate_pad,
    )
    eval_loader = DataLoader(
        SwingSequenceDataset(sequences, eval_ids),
        batch_size=config.batch_size, shuffle=False, collate_fn=collate_pad,
    )

    class_weights = compute_class_weights(sequences, train_ids, NUM_CLASSES)
    print("Class weights (inverse frequency, TRAIN split only):")
    from .constants import PHASE_ORDER
    for label, w in zip(PHASE_ORDER, class_weights.tolist()):
        print(f"  {label:30s} {w:.3f}")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    history = TrainHistory()
    best_acc = 0.0
    checkpoint_path = Path(checkpoint_path)

    for epoch in range(1, config.n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, class_weights, config.smoothing, device)
        eval_loss, eval_acc = evaluate_epoch(model, eval_loader, class_weights, config.smoothing, device)

        history.train_loss.append(train_loss)
        history.eval_loss.append(eval_loss)
        history.eval_acc.append(eval_acc)

        if eval_acc > best_acc:
            best_acc = eval_acc
            save_checkpoint(model, checkpoint_path)

        if epoch % config.log_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                f"eval_loss {eval_loss:.4f} | eval_acc (raw, per-frame) {eval_acc:.4f}"
            )

    print(f"\nBest raw per-frame eval accuracy: {best_acc:.4f}")
    print(f"Checkpoint -> {checkpoint_path}")
    return model, history


def plot_history(history: TrainHistory, output_path: str | Path, title_suffix: str = "") -> None:
    """Save the loss/accuracy curves -- the first place a divergence between
    train and eval loss shows up."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.train_loss, label="train")
    axes[0].plot(history.eval_loss, label="eval")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].legend(); axes[0].set_title("Loss")

    axes[1].plot(history.eval_acc)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("raw per-frame accuracy")
    axes[1].set_title(f"Eval accuracy {title_suffix}".strip())

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Training curves -> {output_path}")
