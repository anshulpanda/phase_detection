"""
Pose extraction and video metadata.

Wraps the YOLO pose model and ffprobe. Both are heavyweight/external, so they
are isolated here and imported lazily -- the rest of the package (model,
decode, metrics) stays importable without ultralytics or ffmpeg installed.
"""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..constants import NO_PHASE_LABEL, POSE_FEATURE_COLS, VIDEO_EXTENSIONS


def load_pose_model(model_path: str | Path, device: str = "cpu"):
    """Load the YOLO pose checkpoint. Imported lazily -- ultralytics is slow
    to import and is only needed for the pose-extraction stages."""
    from ultralytics import YOLO

    return YOLO(str(model_path)).to(device)


def extract_keypoints(result) -> np.ndarray | None:
    """(17, 3) keypoints for the highest-confidence person, or None.

    Clips are single-player by construction, but crowds/ball-kids do appear in
    frame, so the highest-confidence detection is taken rather than the first.
    """
    if result.keypoints is None or len(result.keypoints) == 0:
        return None
    best = 0
    if result.boxes is not None and len(result.boxes) > 0:
        best = int(result.boxes.conf.cpu().numpy().argmax())
    return result.keypoints.data[best].cpu().numpy()


def extract_pose_for_video(
    video_path: Path,
    pose_model,
    batch_size: int = 16,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run pose estimation over every frame, one row per frame.

    Frames are pushed through the model in batches rather than one call per
    frame -- that is the single biggest speed lever in this pipeline.

    Frames with no detected person get NaN-filled keypoint columns; deciding
    what to do about those is the caller's job (see
    `features.handle_missed_detections`).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rows: list[dict] = []
    frame_buffer: list = []
    n_missed = 0
    frame_idx = 0

    def flush_batch():
        nonlocal n_missed
        if not frame_buffer:
            return
        results = pose_model(frame_buffer, verbose=False)
        for offset, result in enumerate(results):
            idx = frame_idx - len(frame_buffer) + offset
            kpts = extract_keypoints(result)
            row = {
                "video_id": video_path.stem,
                "frame_idx": idx,
                "timestamp_s": round(idx / fps, 6) if fps else 0.0,
                "fps": fps,
                "frame_w": frame_w,
                "frame_h": frame_h,
            }
            if kpts is not None:
                for col, val in zip(POSE_FEATURE_COLS, kpts.flatten()):
                    row[col] = float(val)
            else:
                n_missed += 1
                for col in POSE_FEATURE_COLS:
                    row[col] = np.nan
            rows.append(row)
        frame_buffer.clear()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_buffer.append(frame)
        frame_idx += 1
        if len(frame_buffer) >= batch_size:
            flush_batch()
    flush_batch()
    cap.release()

    if n_missed and verbose:
        print(f"  [{video_path.stem}] {n_missed}/{frame_idx} frames had no person detected")

    df = pd.DataFrame(rows)
    df.attrs["total_frames"] = total
    df.attrs["n_missed_detections"] = n_missed
    return df


def get_video_metadata(video_path: Path) -> dict:
    """fps, frame count, duration, chapters and slow-motion flag, via one ffprobe call.

    Chapters are the hand-authored phase boundaries embedded in the labelled
    exports. `is_slowmo` comes from a format tag on the container -- it is
    metadata written by the capture device, not inferred from the frame rate.
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_chapters",
        "-show_format",
        "-show_entries", "stream=avg_frame_rate,nb_frames,duration",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    video_stream = next((s for s in data.get("streams", []) if "avg_frame_rate" in s), {})
    num, den = (video_stream.get("avg_frame_rate", "0/1")).split("/")
    fps = float(num) / float(den) if float(den) != 0 else 0.0

    duration = float(data.get("format", {}).get("duration", 0.0))
    nb_frames = video_stream.get("nb_frames")
    total_frames = int(nb_frames) if nb_frames else int(round(duration * fps))

    chapters = []
    for ch in data.get("chapters", []):
        title = ch.get("tags", {}).get("title", "").strip()
        chapters.append({
            "chapter_id": ch["id"],
            "start_s": float(ch["start_time"]),
            "end_s": float(ch["end_time"]),
            "label": title if title else NO_PHASE_LABEL,
        })

    format_tags = data.get("format", {}).get("tags", {})
    is_slowmo = "SLOWMO" in " ".join(format_tags.values()).upper()

    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_s": duration,
        "chapters": chapters,
        "is_slowmo": is_slowmo,
    }


def label_for_frame(frame_idx: int, fps: float, chapters: list) -> tuple:
    """Phase label and owning chapter for a frame index.

    The final frame can land exactly on the last chapter's end timestamp, which
    a half-open interval test would miss -- hence the explicit tail case.
    """
    t = frame_idx / fps
    for ch in chapters:
        if ch["start_s"] <= t < ch["end_s"]:
            return ch["label"], ch
    if chapters and t >= chapters[-1]["end_s"] - (1 / fps):
        return chapters[-1]["label"], chapters[-1]
    fallback = {"chapter_id": -1, "start_s": 0.0, "end_s": 0.0, "label": NO_PHASE_LABEL}
    return NO_PHASE_LABEL, fallback


def resolve_video_paths(inputs: list) -> list:
    """Expand CLI --videos entries into a deduped, sorted list of video files.

    Each entry may be a directory (scanned recursively), a single video file,
    or a .txt file listing one path per line. Sorting keeps sharding
    deterministic across processes.
    """
    paths: list[Path] = []
    for entry in inputs:
        p = Path(entry)
        if p.is_dir():
            paths.extend(sorted(f for f in p.rglob("*") if f.suffix in VIDEO_EXTENSIONS))
        elif p.suffix == ".txt":
            for line in p.read_text().splitlines():
                line = line.strip()
                if line:
                    paths.append(Path(line))
        elif p.suffix in VIDEO_EXTENSIONS:
            paths.append(p)
        else:
            raise ValueError(f"Don't know how to interpret --videos entry: {entry}")

    seen, unique = set(), []
    for p in sorted(paths):
        if p.resolve() not in seen:
            seen.add(p.resolve())
            unique.append(p)
    return unique


def excluded_stems(exclude_dirs: list, exclude_ids_csv: list, strip_suffix: str = "") -> set:
    """Video ids to skip, gathered from labelled-video directories and/or CSVs.

    `strip_suffix` handles the ground-truth naming convention where labelled
    exports carry a trailing marker (e.g. " CHAPTERED") that the pool copy
    does not.
    """
    stems: set[str] = set()
    for d in exclude_dirs or []:
        for f in Path(d).rglob("*"):
            if f.suffix in VIDEO_EXTENSIONS:
                stem = f.stem
                if strip_suffix and stem.endswith(strip_suffix):
                    stem = stem[: -len(strip_suffix)]
                stems.add(stem)
    for csv_path in exclude_ids_csv or []:
        ids = pd.read_csv(csv_path)["video_id"].astype(str)
        if strip_suffix:
            ids = ids.str.removesuffix(strip_suffix)
        stems.update(ids)
    return stems
