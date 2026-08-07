"""
Confidence scoring and review banding for the active-learning loop.

Layers, innermost first:

  per-frame       softmax probability of the class the monotonic decode chose
                  (see decode.per_frame_confidence)
  per-phase       mean of those, over the frames assigned to that phase
  per-video       min and mean across the 7 per-phase values
  composite       0.6 * min + 0.4 * mean

The composite is deliberately weighted toward the *weakest* phase: a video
should not be able to hide one badly-predicted phase behind several strong
ones, since a single broken phase makes the whole labelled swing unusable.
"""

import re

import numpy as np
import pandas as pd

from ..constants import PHASE_COL_PREFIXES, PHASE_ORDER

COMPOSITE_MIN_WEIGHT = 0.6
COMPOSITE_MEAN_WEIGHT = 0.4

# Fixed absolute bands, as used from round 2 onward:
#   (name, lower bound exclusive, upper bound inclusive, default sample size)
# Round 1 instead searched for a "natural breakpoint" in percentile windows,
# which made bands shift with the pool's distribution shape -- not a stable
# basis for round-over-round comparison, hence the move to fixed thresholds.
BAND_SPECS = [
    ("very_low", 0.00, 0.35, 15),
    ("low_mid", 0.35, 0.55, 15),
    ("mid", 0.55, 0.65, 15),
    ("mid_high", 0.65, 0.75, 12),
    ("high", 0.75, 1.01, 9),
]


def extract_player(video_id: str) -> str:
    """'ALCARAZ_FH (10)' -> 'ALCARAZ'."""
    match = re.match(r"^([A-Za-z]+)", str(video_id))
    return match.group(1) if match else str(video_id)


def add_composite_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Add min/mean/composite confidence from the per-phase columns.

    Note this *overwrites* `min_confidence`/`mean_confidence`, which arrive
    from the prediction CSV as frame-level statistics over the whole video.
    The originals are preserved as `video_frame_*` because the two mean
    genuinely different things and both are worth keeping.
    """
    conf_cols = [f"{p}_confidence" for p in PHASE_COL_PREFIXES]
    conf_matrix = df[conf_cols].to_numpy(dtype=float)

    # A video where no phase was detected at all is an all-NaN slice; numpy
    # warns rather than failing, and the resulting NaN composite is correct.
    with np.errstate(invalid="ignore"):
        min_conf = np.nanmin(conf_matrix, axis=1)
        mean_conf = np.nanmean(conf_matrix, axis=1)

    if "min_confidence" in df.columns:
        df["video_frame_min_confidence"] = df["min_confidence"]
    if "mean_confidence" in df.columns:
        df["video_frame_mean_confidence"] = df["mean_confidence"]

    df["min_confidence"] = min_conf
    df["mean_confidence"] = mean_conf
    df["composite_confidence"] = COMPOSITE_MIN_WEIGHT * min_conf + COMPOSITE_MEAN_WEIGHT * mean_conf
    return df


def add_missing_phase_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flag phases the decode never emitted anywhere in a video.

    A missing phase is a stronger red flag than low confidence: the model
    produced no prediction for that phase at all, rather than an unsure one.
    `only_no_phase_missing` is the benign case -- the clip simply never had an
    idle frame, not a broken swing decode.
    """
    missing_mask = pd.DataFrame({
        prefix: df[f"{prefix}_start_frame"].isna() for prefix in PHASE_COL_PREFIXES
    })
    df["n_missing_phases"] = missing_mask.sum(axis=1)
    df["missing_phases"] = missing_mask.apply(
        lambda row: [phase for phase, missing in zip(PHASE_ORDER, row) if missing], axis=1
    )
    df["only_no_phase_missing"] = df["missing_phases"].apply(lambda phases: phases == ["no_phase"])
    return df


def get_real_missing_phase_videos(df: pd.DataFrame) -> pd.DataFrame:
    """Videos missing at least one real (non-`no_phase`) phase."""
    with_missing = df[df["n_missing_phases"] > 0]
    return with_missing[~with_missing["only_no_phase_missing"]]


def has_real_missing_phase(pred_row: pd.Series) -> bool:
    """Row-level equivalent of the above, for a single prediction row."""
    for prefix in PHASE_COL_PREFIXES:
        if prefix == "no_phase":
            continue
        if pd.isna(pred_row.get(f"{prefix}_start_frame")):
            return True
    return False


def assign_bands(df: pd.DataFrame, band_specs: list | None = None) -> pd.DataFrame:
    """Label each row with its fixed confidence band."""
    band_specs = band_specs or BAND_SPECS
    df["confidence_band"] = None
    for name, lo, hi, _ in band_specs:
        in_band = (df["composite_confidence"] > lo) & (df["composite_confidence"] <= hi)
        df.loc[in_band, "confidence_band"] = name
    return df


def stratified_sample_by_player(
    band_df: pd.DataFrame, n_total: int, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.Series, list]:
    """Sample `n_total` videos from a band, proportionally across players.

    Uses largest-remainder rounding so the per-player quotas sum exactly to
    `n_total`, capped at each player's availability. When a player has fewer
    videos than their quota, everything they have is taken and the shortfall
    redistributes -- recorded in the returned fallback list.

    Returns (sampled_df, per_player_allocation, fallback_players).
    """
    counts = band_df["player"].value_counts()
    total = len(band_df)
    if total == 0:
        return band_df.iloc[0:0], pd.Series(dtype=int), []
    n_total = min(n_total, total)

    quotas = n_total * counts / total
    base = np.minimum(np.floor(quotas).astype(int), counts)
    remaining = n_total - int(base.sum())

    # Largest-remainder pass: hand out leftovers to the biggest fractional parts.
    remainders = quotas - base
    capacity = counts - base
    order = remainders[capacity > 0].sort_values(ascending=False).index.tolist()

    i = 0
    while remaining > 0 and i < len(order):
        player = order[i]
        if counts[player] - base[player] > 0:
            base[player] += 1
            remaining -= 1
        i += 1

    # Round-robin mop-up for any residue left by capacity caps.
    guard = 0
    while remaining > 0 and guard < 10000:
        for player in counts.index:
            if remaining <= 0:
                break
            if counts[player] - base[player] > 0:
                base[player] += 1
                remaining -= 1
        guard += 1

    sampled_parts, fallback_players = [], []
    for player, n in base.items():
        if n <= 0:
            continue
        pool = band_df[band_df["player"] == player]
        if n >= len(pool):
            sampled_parts.append(pool)
            fallback_players.append((player, len(pool)))
        else:
            sampled_parts.append(pool.sample(n=n, random_state=int(rng.integers(0, 1_000_000))))

    sampled = pd.concat(sampled_parts) if sampled_parts else band_df.iloc[0:0]
    return sampled, base, fallback_players


def plot_confidence_distribution(pool_df: pd.DataFrame, output_path, band_specs: list | None = None) -> None:
    """Histogram of composite confidence with band cutoffs marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    band_specs = band_specs or BAND_SPECS
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pool_df["composite_confidence"].dropna(), bins=40, color="#6b8fb8", edgecolor="white")
    colors = ["#d1495b", "#e3a72c", "#8a8a8a", "#4c8577", "#2a9d8f"]
    for (name, _lo, hi, _), color in zip(band_specs, colors):
        ax.axvline(hi, color=color, linestyle="--", linewidth=1.5, label=f"{name} upper = {hi:.2f}")
    ax.set_xlabel("composite_confidence")
    ax.set_ylabel("# videos")
    ax.set_title(f"Composite confidence distribution ({len(pool_df)}-video pool)")
    ax.legend(fontsize=8)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Confidence distribution -> {output_path}")
