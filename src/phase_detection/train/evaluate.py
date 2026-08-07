"""
Evaluation metrics.

The headline metric is swing-phase **boundary error in milliseconds**: for each
phase, how far off is the predicted first/last frame from the labelled one.
Milliseconds rather than frames because a 10-frame error at 30fps is twice the
real-world gap of the same error at 60fps, and the dataset mixes frame rates.

Aggregation note (this matters and is easy to get wrong): overall mean, median
and standard deviation are computed by **pooling every individual start- and
end-error measurement into one flat distribution**, then taking statistics of
that. They are not averages of the per-phase statistics. Medians and standard
deviations do not average -- the combined median depends on the sorted order of
every value, and combined variance includes a between-group term that averaging
per-phase spreads discards entirely. Only a size-weighted mean would survive
that shortcut, so pooling raw values is done for all three.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..constants import PHASE_ORDER
from ..decode import get_phase_boundaries, monotonic_decode


@torch.no_grad()
def evaluate_boundaries(
    model,
    sequences: dict,
    video_ids: list,
    phase_order: list | None = None,
    device: torch.device | str = "cpu",
    allow_no_phase_reset: bool = True,
) -> pd.DataFrame:
    """Per-(video, phase) boundary error, in frames and seconds.

    `missed_in_pred` marks a phase the decode never emitted. Those rows carry
    NaN errors and are dropped from the statistics below -- worth remembering
    when reading the numbers, since a total miss is arguably worse than a large
    offset but does not count against the mean.
    """
    phase_order = phase_order or PHASE_ORDER
    model.eval()
    rows = []

    for vid in video_ids:
        item = sequences[vid]
        fps = item["fps"]

        feats = torch.from_numpy(item["features"]).float().unsqueeze(0).to(device)
        probs = torch.softmax(model(feats), dim=-1).squeeze(0).cpu().numpy()

        decoded = monotonic_decode(probs, phase_order, allow_no_phase_reset=allow_no_phase_reset)

        true_b = get_phase_boundaries(item["labels"], phase_order)
        pred_b = get_phase_boundaries(decoded, phase_order)

        for phase_name in true_b:
            t_start, t_end = true_b[phase_name]["start"], true_b[phase_name]["end"]
            p_start, p_end = pred_b[phase_name]["start"], pred_b[phase_name]["end"]

            start_f = None if (t_start is None or p_start is None) else abs(p_start - t_start)
            end_f = None if (t_end is None or p_end is None) else abs(p_end - t_end)

            rows.append({
                "video_id": vid,
                "player": item["player"],
                "fps": fps,
                "phase": phase_name,
                "start_error_frames": start_f,
                "start_error_seconds": None if start_f is None else start_f / fps,
                "end_error_frames": end_f,
                "end_error_seconds": None if end_f is None else end_f / fps,
                "missed_in_pred": p_start is None,
            })

    return pd.DataFrame(rows)


def pooled_error_stats_ms(boundary_df: pd.DataFrame) -> pd.Series:
    """Mean/median/variance/std over every start and end error, pooled.

    See the module docstring on why this pools raw values rather than combining
    per-phase summaries.
    """
    pooled = pd.concat([
        boundary_df["start_error_seconds"].dropna(),
        boundary_df["end_error_seconds"].dropna(),
    ]) * 1000
    if pooled.empty:
        return pd.Series({"n": 0, "mean_ms": np.nan, "median_ms": np.nan,
                          "var_ms2": np.nan, "std_ms": np.nan, "max_ms": np.nan})
    return pd.Series({
        "n": len(pooled),
        "mean_ms": pooled.mean(),
        "median_ms": pooled.median(),
        "var_ms2": pooled.var(),
        "std_ms": pooled.std(),
        "max_ms": pooled.max(),
    })


def per_phase_error_table(boundary_df: pd.DataFrame, phase_order: list | None = None) -> pd.DataFrame:
    """Per-phase mean/median, both for start and end boundaries and pooled.

    `pooled_*` columns combine that phase's start and end errors -- the
    per-phase analogue of the overall figure.
    """
    phase_order = phase_order or PHASE_ORDER
    rows = []
    for phase_name in phase_order:
        if phase_name == "no_phase":
            continue
        sub = boundary_df[boundary_df["phase"] == phase_name]
        ss = sub["start_error_seconds"].dropna() * 1000
        es = sub["end_error_seconds"].dropna() * 1000
        pooled = pd.concat([ss, es])
        rows.append({
            "phase": phase_name,
            "n_videos": len(sub),
            "n_missed": int(sub["missed_in_pred"].sum()),
            "start_mean_ms": ss.mean() if len(ss) else np.nan,
            "start_median_ms": ss.median() if len(ss) else np.nan,
            "end_mean_ms": es.mean() if len(es) else np.nan,
            "end_median_ms": es.median() if len(es) else np.nan,
            "pooled_mean_ms": pooled.mean() if len(pooled) else np.nan,
            "pooled_median_ms": pooled.median() if len(pooled) else np.nan,
        })
    return pd.DataFrame(rows)


def per_player_error_table(boundary_df: pd.DataFrame) -> pd.DataFrame:
    """Pooled error by player -- surfaces whether a single held-out player
    is dragging the aggregate."""
    rows = []
    for player, sub in boundary_df.groupby("player"):
        pooled = pd.concat([
            sub["start_error_seconds"].dropna(),
            sub["end_error_seconds"].dropna(),
        ]) * 1000
        if pooled.empty:
            continue
        rows.append({
            "player": player,
            "n": len(pooled),
            "mean_ms": pooled.mean(),
            "median_ms": pooled.median(),
            "max_ms": pooled.max(),
        })
    return pd.DataFrame(rows).sort_values("mean_ms", ascending=False).reset_index(drop=True)


def train_vs_eval_report(
    model,
    sequences: dict,
    train_ids: list,
    eval_ids: list,
    device: torch.device | str = "cpu",
    allow_no_phase_reset: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pooled boundary error on both splits, to expose the overfitting gap.

    A test error much larger than train error is the signature to look for;
    the loss curves show the same thing in abstract units, this shows it in
    milliseconds of real timing error at the selected checkpoint.
    """
    train_df = evaluate_boundaries(model, sequences, train_ids, device=device,
                                    allow_no_phase_reset=allow_no_phase_reset)
    eval_df = evaluate_boundaries(model, sequences, eval_ids, device=device,
                                   allow_no_phase_reset=allow_no_phase_reset)
    summary = pd.DataFrame({
        "train": pooled_error_stats_ms(train_df),
        "eval": pooled_error_stats_ms(eval_df),
    })
    return summary, train_df, eval_df


@torch.no_grad()
def frame_accuracy(
    model,
    sequences: dict,
    video_ids: list,
    device: torch.device | str = "cpu",
    allow_no_phase_reset: bool = True,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Raw (per-frame argmax) and decoded accuracy over a split.

    Decode usually helps by removing flicker, but it can also propagate an
    early mistake forward -- it cannot undo a wrong monotonic step except via
    the no_phase reset -- so both numbers are reported.

    Returns (raw_acc, decoded_acc, all_true, all_decoded).
    """
    model.eval()
    raw_preds_all, decoded_all, true_all = [], [], []

    for vid in video_ids:
        item = sequences[vid]
        feats = torch.from_numpy(item["features"]).float().unsqueeze(0).to(device)
        probs = torch.softmax(model(feats), dim=-1).squeeze(0).cpu().numpy()

        raw_preds_all.append(probs.argmax(axis=-1))
        decoded_all.append(monotonic_decode(probs, PHASE_ORDER, allow_no_phase_reset=allow_no_phase_reset))
        true_all.append(item["labels"])

    raw = np.concatenate(raw_preds_all)
    decoded = np.concatenate(decoded_all)
    true = np.concatenate(true_all)
    return float((raw == true).mean()), float((decoded == true).mean()), true, decoded


def classification_report_text(true: np.ndarray, pred: np.ndarray) -> str:
    """Per-class precision/recall/F1 -- shows whether the rare short phases are
    actually being learned, which a single accuracy number hides."""
    from sklearn.metrics import classification_report

    return classification_report(true, pred, target_names=PHASE_ORDER,
                                  labels=list(range(len(PHASE_ORDER))), zero_division=0)


def plot_confusion_matrix(true: np.ndarray, pred: np.ndarray, output_path: str | Path) -> None:
    """Confusion matrix. Because labels are ordinal, the thing to look for is
    whether off-diagonal mass sits *near* the diagonal (adjacent-phase
    confusion, expected at transitions) or far from it (a real failure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(true, pred, labels=list(range(len(PHASE_ORDER))))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(PHASE_ORDER)), PHASE_ORDER, rotation=45, ha="right")
    ax.set_yticks(range(len(PHASE_ORDER)), PHASE_ORDER)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (decoded)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix -> {output_path}")
