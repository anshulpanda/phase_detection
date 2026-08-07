"""
Loss construction: ordinal label smoothing, class weighting, masked soft CE.

Three things are being corrected for simultaneously:

  Ordinal structure -- phase transitions are genuinely continuous, not hard
  boundaries, so a frame's target distributes some mass to its immediate
  ordinal neighbours rather than being one-hot. Frames near a transition get
  partial credit for both sides.

  Class imbalance -- `no_phase` and `Follow Through` together are ~65% of all
  frames while `Contact` is ~3%. Without inverse-frequency weighting the model
  has little incentive to learn the short, rare, and most interesting phases.

  Variable length -- sequences are padded to the batch maximum; padded
  positions must not contribute to the loss.
"""

import numpy as np
import torch

from ..constants import NUM_CLASSES

PAD_LABEL = -100


def build_ordinal_smoothed_targets(
    labels: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """Soft targets where ordinal neighbours share the smoothing mass.

    The true class keeps (1 - smoothing); its immediate neighbours in
    PHASE_ORDER split the remainder. `no_phase` sits at index 0, so it is
    treated as the ordinal neighbour of `Start of Unit Turn` *for smoothing
    purposes* -- frames near the no_phase -> swing-start boundary get partial
    credit for both.

    Args:
        labels: (batch, time) int64, may contain PAD_LABEL for padding.

    Returns:
        (batch, time, num_classes); padded positions are all-zero.
    """
    batch, time = labels.shape
    valid = labels != PAD_LABEL
    # Placeholder index for padded positions; zeroed out via `valid` at the end.
    safe_labels = labels.clamp(min=0)

    targets = torch.zeros(batch, time, num_classes)
    targets.scatter_(-1, safe_labels.unsqueeze(-1), 1.0 - smoothing)

    has_lower = safe_labels > 0
    has_upper = safe_labels < (num_classes - 1)
    n_neighbors = (has_lower.long() + has_upper.long()).clamp(min=1)
    share = smoothing / n_neighbors.float()

    lower_idx = (safe_labels - 1).clamp(min=0)
    upper_idx = (safe_labels + 1).clamp(max=num_classes - 1)

    zeros = torch.zeros_like(share)
    targets.scatter_add_(-1, lower_idx.unsqueeze(-1), torch.where(has_lower, share, zeros).unsqueeze(-1))
    targets.scatter_add_(-1, upper_idx.unsqueeze(-1), torch.where(has_upper, share, zeros).unsqueeze(-1))

    # Unreachable while num_classes > 1, but keeps the distribution normalized
    # if the class list is ever reduced to a single entry.
    no_neighbor = (~has_lower) & (~has_upper)
    if no_neighbor.any():
        targets.scatter_add_(
            -1,
            safe_labels.unsqueeze(-1),
            torch.where(no_neighbor, torch.tensor(smoothing), torch.zeros(1)).unsqueeze(-1),
        )

    return targets * valid.unsqueeze(-1).float()


def compute_class_weights(
    sequences: dict,
    train_ids: list,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    """Inverse-frequency class weights, computed on the TRAIN split only.

    Computing these over the full dataset would leak test-set label
    distribution into training.
    """
    counts = np.zeros(num_classes)
    for v in train_ids:
        labels = sequences[v]["labels"]
        for c in range(num_classes):
            counts[c] += (labels == c).sum()
    counts = np.maximum(counts, 1)  # never divide by zero on an absent class
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def soft_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy against soft targets, masked over padding.

    Class weighting is applied via the argmax of the soft target (i.e. the
    frame's dominant/true class), so smoothing does not blur which weight a
    frame receives.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    per_frame_loss = -(soft_targets * log_probs).sum(dim=-1)

    if class_weights is not None:
        hard_labels = soft_targets.argmax(dim=-1)
        per_frame_loss = per_frame_loss * class_weights[hard_labels]

    per_frame_loss = per_frame_loss * mask.float()
    return per_frame_loss.sum() / mask.float().sum().clamp(min=1)
