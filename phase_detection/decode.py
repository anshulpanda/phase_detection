"""
Constrained sequence decoding and boundary extraction.

The TCN emits an independent probability distribution per frame. On its own,
a per-frame argmax flickers at phase transitions and can produce physically
impossible sequences (a swing that goes backwards). `monotonic_decode` is a
Viterbi-style dynamic program that instead finds the single highest-scoring
*legal* path across the whole clip.

Transition grammar, per step:
  - stay in the current phase
  - advance exactly one step forward through PHASE_ORDER
  - optionally, enter any phase from `no_phase` (see `allow_no_phase_reset`)

`allow_no_phase_reset` is a real behavioural fork that existed in the original
notebooks, kept here explicitly rather than silently unified:

  True  (training/evaluation default) -- any phase is reachable directly from
        `no_phase`. Lets the decode snap back to idle and re-enter, which
        matches how the training notebooks measured boundary error.

  False (prediction default) -- the reset edge is removed, which guarantees
        every phase gets a contiguous span rather than being skipped over.
        The prediction pipeline relies on this: a phase that never appears is
        a meaningful "missing phase" signal used downstream by the review
        tooling, so the decode must not be able to route around a phase.

Changing this flag changes the numbers. Match it to whichever convention the
artifact you are comparing against used.
"""

import numpy as np

from .constants import PHASE_ORDER


def monotonic_decode(
    probs: np.ndarray,
    phase_order: list | None = None,
    allow_no_phase_reset: bool = True,
) -> np.ndarray:
    """Best legal phase path through per-frame probabilities.

    Args:
        probs: (T, C) per-frame class probabilities (softmax output).
        phase_order: class list; defaults to PHASE_ORDER.
        allow_no_phase_reset: whether `no_phase` -> any phase is a legal
            transition. See the module docstring -- this is not a tuning knob.

    Returns:
        (T,) array of class indices.
    """
    phase_order = phase_order or PHASE_ORDER
    num_classes = probs.shape[1]
    n_frames = probs.shape[0]
    no_phase_id = phase_order.index("no_phase")

    # Work in log space so path scores add rather than multiply (underflow).
    log_probs = np.log(np.clip(probs, 1e-8, 1.0))

    dp = np.full((n_frames, num_classes), -np.inf)
    backptr = np.zeros((n_frames, num_classes), dtype=np.int64)
    dp[0] = log_probs[0]

    for t in range(1, n_frames):
        for c in range(num_classes):
            candidates = [c]                      # stay
            if c > 0:
                candidates.append(c - 1)          # advance one ordinal step
            if allow_no_phase_reset and c != no_phase_id:
                candidates.append(no_phase_id)    # enter from idle
            best_prev = max(candidates, key=lambda p: dp[t - 1, p])
            dp[t, c] = dp[t - 1, best_prev] + log_probs[t, c]
            backptr[t, c] = best_prev

    path = np.zeros(n_frames, dtype=np.int64)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(n_frames - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]
    return path


def get_phase_boundaries(phase_sequence: np.ndarray, phase_order: list | None = None) -> dict:
    """First and last frame index of each phase.

    Returns {phase_name: {"start": int|None, "end": int|None}}, with None when
    the phase never appears in the sequence. `no_phase` is excluded -- it is
    background, not a boundary of interest.
    """
    phase_order = phase_order or PHASE_ORDER
    boundaries = {}
    for phase_name in phase_order:
        if phase_name == "no_phase":
            continue
        phase_id = phase_order.index(phase_name)
        frames = np.where(phase_sequence == phase_id)[0]
        if len(frames):
            boundaries[phase_name] = {"start": int(frames[0]), "end": int(frames[-1])}
        else:
            boundaries[phase_name] = {"start": None, "end": None}
    return boundaries


def extract_swing_boundaries(phase_sequence: np.ndarray, phase_order: list | None = None) -> tuple:
    """First/last frame that is not `no_phase` -- i.e. the swing itself.

    Returns (None, None) when no swing was detected at all.
    """
    phase_order = phase_order or PHASE_ORDER
    no_phase_id = phase_order.index("no_phase")
    swing_frames = np.where(phase_sequence != no_phase_id)[0]
    if len(swing_frames) == 0:
        return None, None
    return int(swing_frames[0]), int(swing_frames[-1])


def per_frame_confidence(probs: np.ndarray, decoded: np.ndarray) -> np.ndarray:
    """Probability the model assigned to the class the decoder actually chose.

    Deliberately not `probs.max(axis=-1)`: at an ambiguous transition frame the
    decoder can override the raw argmax to keep the sequence legal, and the
    reported confidence should reflect trust in the emitted label, not in an
    unconstrained per-frame guess.
    """
    return probs[np.arange(len(decoded)), decoded]


def per_phase_confidence(
    probs: np.ndarray, decoded: np.ndarray, phase_order: list | None = None
) -> dict:
    """Mean decoded-class confidence over the frames assigned to each phase.

    A phase the decoder never emitted maps to None rather than 0.0 -- absent
    and unconfident are different signals downstream.
    """
    phase_order = phase_order or PHASE_ORDER
    frame_conf = per_frame_confidence(probs, decoded)
    out = {}
    for phase_id, phase_name in enumerate(phase_order):
        mask = decoded == phase_id
        out[phase_name] = float(frame_conf[mask].mean()) if mask.any() else None
    return out
