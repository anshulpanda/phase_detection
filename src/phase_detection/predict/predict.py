"""
Inference over unlabeled videos.

Full pipeline per video: pose extraction -> short-gap interpolation ->
hip/torso normalization -> TCN -> monotonic decode -> per-phase boundaries and
confidences. Output matches the wide predictions-CSV schema the review tooling
consumes.

Note the decode here defaults to `allow_no_phase_reset=False`. Without the
reset edge, every phase is guaranteed a contiguous span, so a phase that comes
back missing genuinely was never predicted -- which is exactly the signal the
review stage keys on. See decode.py.
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..constants import PHASE_ORDER, PHASE_TO_ID
from ..decode import monotonic_decode, per_frame_confidence, per_phase_confidence
from ..generate_dataset.features import handle_missed_detections, normalize_pose
from ..pose_estimation.pose import extract_pose_for_video

OUTPUT_COLUMNS = ["video_id", "video_path", "fps", "n_frames", "mean_confidence", "min_confidence"]
for _phase in PHASE_ORDER:
    _prefix = _phase.replace(" ", "_")
    OUTPUT_COLUMNS += [
        f"{_prefix}_start_frame", f"{_prefix}_start_timestamp_s",
        f"{_prefix}_end_frame", f"{_prefix}_end_timestamp_s",
        f"{_prefix}_confidence",
    ]

FAILURE_COLUMNS = ["video_id", "video_path", "error"]


def predict_video(
    video_path: Path,
    model,
    pose_model,
    device: torch.device | str = "cpu",
    batch_size: int = 16,
    max_missed_frac: float = 0.3,
    allow_no_phase_reset: bool = False,
) -> dict:
    """Run the full inference pipeline on one video."""
    vdf = extract_pose_for_video(video_path, pose_model, batch_size)
    vdf = vdf.sort_values("frame_idx")
    vdf = handle_missed_detections(vdf, max_missed_frac)

    fps = float(vdf["fps"].iloc[0])
    features = normalize_pose(vdf)

    with torch.no_grad():
        feats_t = torch.from_numpy(features).float().unsqueeze(0).to(device)
        probs = torch.softmax(model(feats_t), dim=-1).squeeze(0).cpu().numpy()

    decoded = monotonic_decode(probs, PHASE_ORDER, allow_no_phase_reset=allow_no_phase_reset)
    frame_conf = per_frame_confidence(probs, decoded)

    return {
        "video_id": video_path.stem,
        "video_path": str(video_path),
        "fps": fps,
        "n_frames": len(decoded),
        "decoded_phases": decoded,
        "mean_confidence": float(frame_conf.mean()),
        "min_confidence": float(frame_conf.min()),
        "per_phase_confidence": per_phase_confidence(probs, decoded, PHASE_ORDER),
    }


def row_from_result(result: dict) -> dict:
    """Flatten a prediction result into the wide CSV schema."""
    fps, decoded = result["fps"], result["decoded_phases"]
    row = {
        "video_id": result["video_id"],
        "video_path": result["video_path"],
        "fps": fps,
        "n_frames": result["n_frames"],
        "mean_confidence": result["mean_confidence"],
        "min_confidence": result["min_confidence"],
    }
    for phase_name in PHASE_ORDER:
        prefix = phase_name.replace(" ", "_")
        frames = np.where(decoded == PHASE_TO_ID[phase_name])[0]
        if len(frames):
            start_frame, end_frame = int(frames[0]), int(frames[-1])
            row[f"{prefix}_start_frame"] = start_frame
            row[f"{prefix}_start_timestamp_s"] = round(start_frame / fps, 6)
            row[f"{prefix}_end_frame"] = end_frame
            row[f"{prefix}_end_timestamp_s"] = round(end_frame / fps, 6)
        else:
            row[f"{prefix}_start_frame"] = None
            row[f"{prefix}_start_timestamp_s"] = None
            row[f"{prefix}_end_frame"] = None
            row[f"{prefix}_end_timestamp_s"] = None
        row[f"{prefix}_confidence"] = result["per_phase_confidence"].get(phase_name)
    return row


def already_done_ids(output_path: Path) -> set:
    if not Path(output_path).exists():
        return set()
    try:
        return set(pd.read_csv(output_path)["video_id"].astype(str))
    except Exception:
        return set()


def predict_batch(
    videos: list,
    model,
    pose_model,
    output_path: Path,
    device: torch.device | str = "cpu",
    batch_size: int = 16,
    max_missed_frac: float = 0.3,
    resume: bool = False,
    allow_no_phase_reset: bool = False,
    progress_every: int = 25,
) -> tuple[int, int]:
    """Predict over many videos, writing incrementally.

    Rows are flushed per video so a long run can be interrupted and resumed
    (`--resume`) without losing work. A video that fails -- most often because
    too many frames had no detected person -- is recorded in a sibling
    `*_failures.csv` and does not abort the run.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path = output_path.with_name(output_path.stem + "_failures.csv")

    write_header = not (resume and output_path.exists())
    fail_write_header = not (resume and failures_path.exists())
    mode = "a" if resume else "w"

    n_ok, n_failed = 0, 0

    with open(output_path, mode, newline="") as out_f, open(failures_path, mode, newline="") as fail_f:
        out_writer = csv.DictWriter(out_f, fieldnames=OUTPUT_COLUMNS)
        fail_writer = csv.DictWriter(fail_f, fieldnames=FAILURE_COLUMNS)
        if write_header:
            out_writer.writeheader()
        if fail_write_header:
            fail_writer.writeheader()

        for i, vp in enumerate(videos):
            try:
                result = predict_video(
                    vp, model, pose_model, device, batch_size, max_missed_frac, allow_no_phase_reset
                )
                out_writer.writerow(row_from_result(result))
                out_f.flush()
                n_ok += 1
            except Exception as e:
                fail_writer.writerow({"video_id": vp.stem, "video_path": str(vp), "error": str(e)})
                fail_f.flush()
                n_failed += 1
                print(f"[{i + 1}/{len(videos)}] FAILED {vp.stem}: {e}")

            if progress_every and (i + 1) % progress_every == 0:
                print(f"[{i + 1}/{len(videos)}] processed  (ok={n_ok}, failed={n_failed})")

    print(f"\nDone. {n_ok} succeeded, {n_failed} failed.")
    print(f"Predictions -> {output_path}")
    if n_failed:
        print(f"Failures    -> {failures_path}")
    return n_ok, n_failed
