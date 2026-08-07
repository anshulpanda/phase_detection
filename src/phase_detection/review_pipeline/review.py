"""
Active-learning review resolution.

Turns one round's human audit back into training labels. Three sources feed the
next round's video index:

  verdict == "correct"
      The model's own predicted boundaries are promoted to pseudo-ground-truth
      unchanged. Frames come from the original pool video.

  verdict in {partial, incorrect, labeled}
      The reviewer re-cut the chapters and exported a corrected file; its
      chapter marks are read back with ffprobe. The audit CSV's `band` column
      is *not* trusted to locate that file -- corrected exports did not
      reliably land in the folder matching their band -- so every correction
      directory is searched by filename.

  unreviewed, above the confidence threshold
      Auto-accepted without a human pass. The threshold is empirical, not
      arbitrary: round 1's audited baseline showed ~100% verdict=="correct"
      above composite confidence 0.55. Videos with a genuinely missing phase
      are never auto-accepted regardless of how confident the model was.

A verdict of NaN means "not yet reviewed" and is skipped rather than guessed at.
"""

from pathlib import Path

import pandas as pd

from .confidence import (
    add_composite_confidence,
    extract_player,
    has_real_missing_phase,
)
from ..constants import PHASE_ORDER
from ..pose_estimation.pose import get_video_metadata

DEFAULT_CONFIDENCE_THRESHOLD = 0.55
CORRECTED_FILE_TEMPLATE = "{video_id}_Labelled.mp4"


def find_corrected_file(video_id: str, correction_dirs: list) -> Path | None:
    """Locate a reviewer's corrected export by name across all candidate dirs."""
    for d in correction_dirs:
        candidate = Path(d) / CORRECTED_FILE_TEMPLATE.format(video_id=video_id)
        if candidate.exists():
            return candidate
    return None


def chapters_from_predictions(pred_row: pd.Series, phase_order: list | None = None) -> list | None:
    """Convert wide predicted boundaries into a chapter list.

    Returns None if any real phase is missing -- an incomplete decode cannot
    become ground truth. A missing `no_phase` is tolerated: plenty of clips
    are swing from the first frame.

    `end_s` uses (end_frame + 1) / fps so the chapter spans through the end of
    that frame rather than stopping at its start instant.
    """
    phase_order = phase_order or PHASE_ORDER
    fps = float(pred_row["fps"])
    chapters = []
    for phase in phase_order:
        prefix = phase.replace(" ", "_")
        start_frame = pred_row.get(f"{prefix}_start_frame")
        end_frame = pred_row.get(f"{prefix}_end_frame")
        if pd.isna(start_frame) or pd.isna(end_frame):
            if phase == "no_phase":
                continue
            return None
        chapters.append({
            "chapter_id": len(chapters),
            "start_s": float(start_frame) / fps,
            "end_s": (float(end_frame) + 1) / fps,
            "label": phase,
        })
    return chapters


def _row_from_predictions(video_id, pred_row, meta, label_source, band, verdict, video_dir_name="all"):
    n_frames = float(pred_row["n_frames"])
    fps = float(pred_row["fps"])
    chapters = chapters_from_predictions(pred_row)
    if chapters is None:
        return None
    return {
        "video_id": video_id,
        "video_path": f"{video_dir_name}/{video_id}.mp4",
        "video_fps": fps,
        "video_total_frames": int(n_frames),
        "video_duration_s": n_frames / fps,
        "is_slowmo": bool(meta["is_slowmo"]),
        "chapters": chapters,
        "num_chapters": len(chapters),
        "expected_frames": n_frames,
        "player": extract_player(video_id),
        "label_source": label_source,
        "band": band,
        "verdict": verdict,
    }


def resolve_reviewed(
    review: pd.DataFrame,
    preds: pd.DataFrame,
    pool_meta: pd.DataFrame,
    correction_dirs: list,
    data_root: Path,
) -> tuple[list, list]:
    """Resolve the audited videos into training rows.

    Returns (rows, skipped) where skipped entries are
    (video_id, band, verdict, reason) -- every drop is reported rather than
    silently swallowed, since a systematic skip means a broken round.
    """
    rows, skipped = [], []

    for _, r in review.iterrows():
        video_id = r["video_id"]
        band = r.get("band")
        verdict = r["verdict"] if pd.notna(r.get("verdict")) else None

        if verdict is None:
            skipped.append((video_id, band, verdict, "not yet reviewed (verdict is NaN)"))
            continue

        if verdict == "correct":
            if video_id not in preds.index:
                skipped.append((video_id, band, verdict, "verdict=correct but no prediction row"))
                continue
            if video_id not in pool_meta.index:
                skipped.append((video_id, band, verdict, "no row in video_pool_metadata.csv"))
                continue
            row = _row_from_predictions(
                video_id, preds.loc[video_id], pool_meta.loc[video_id],
                "predicted_pseudo_label", band, verdict,
            )
            if row is None:
                skipped.append((video_id, band, verdict, "predicted boundaries incomplete (NaN column)"))
                continue
            rows.append(row)
            continue

        corrected_file = find_corrected_file(video_id, correction_dirs)
        if corrected_file is None:
            skipped.append((video_id, band, verdict, "no *_Labelled.mp4 export found"))
            continue

        meta = get_video_metadata(corrected_file)
        try:
            rel_path = str(corrected_file.relative_to(data_root))
        except ValueError:
            rel_path = str(corrected_file)

        rows.append({
            "video_id": video_id,
            "video_path": rel_path,
            "video_fps": meta["fps"],
            "video_total_frames": meta["total_frames"],
            "video_duration_s": meta["duration_s"],
            "is_slowmo": meta["is_slowmo"],
            "chapters": meta["chapters"],
            "num_chapters": len(meta["chapters"]),
            "expected_frames": meta["duration_s"] * meta["fps"],
            "player": extract_player(video_id),
            "label_source": "corrected_relabel",
            "band": band,
            "verdict": verdict,
        })

    return rows, skipped


def resolve_auto_accepted(
    preds_all: pd.DataFrame,
    reviewed_ids: set,
    pool_meta: pd.DataFrame,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> tuple[list, list]:
    """Auto-accept unreviewed predictions above the confidence threshold."""
    remaining = preds_all[~preds_all["video_id"].isin(reviewed_ids)].copy()
    if remaining.empty:
        return [], []
    remaining = add_composite_confidence(remaining).set_index("video_id")

    rows, skipped = [], []
    band_label = f"auto_gt_{threshold:.2f}"

    for video_id, pred_row in remaining.iterrows():
        if not (pred_row["composite_confidence"] > threshold):
            continue
        if has_real_missing_phase(pred_row):
            skipped.append((video_id, band_label, None, "real missing phase, not auto-accepted"))
            continue
        if video_id not in pool_meta.index:
            skipped.append((video_id, band_label, None, "no row in video_pool_metadata.csv"))
            continue
        row = _row_from_predictions(
            video_id, pred_row, pool_meta.loc[video_id],
            "predicted_pseudo_label_unreviewed", band_label, None,
        )
        if row is None:
            skipped.append((video_id, band_label, None, "predicted boundaries incomplete (NaN column)"))
            continue
        rows.append(row)

    return rows, skipped
