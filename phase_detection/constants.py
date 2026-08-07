"""
Label vocabulary and pose feature layout.

These values are load-bearing: PHASE_ORDER defines the ordinal class indices
the model is trained against, and POSE_FEATURE_COLS defines the exact column
order of the 51-dim per-frame feature vector. Changing either invalidates
every existing checkpoint, so they live here alone rather than being redefined
per module.
"""

NO_PHASE_LABEL = "no_phase"

# Ordinal phase sequence. `no_phase` sits at index 0 and acts as the "floor"
# class -- it is a real class the model predicts, not a dropped background.
# The ordering matters twice over: ordinal label smoothing treats adjacent
# entries as neighbours, and the monotonic decode only allows forward steps
# through this list.
PHASE_ORDER = [
    "no_phase",
    "Start of Unit Turn",
    "End of Backswing",
    "Forward Swing Initiation",
    "Acceleration",
    "Contact",
    "Follow Through",
]
PHASE_TO_ID = {label: idx for idx, label in enumerate(PHASE_ORDER)}
ID_TO_PHASE = {idx: label for label, idx in PHASE_TO_ID.items()}
NUM_CLASSES = len(PHASE_ORDER)

# Column-name prefixes used in the wide prediction CSVs, e.g. "Contact_start_frame".
PHASE_COL_PREFIXES = [p.replace(" ", "_") for p in PHASE_ORDER]

# COCO-17 keypoint order, as emitted by the YOLO pose model.
KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# 17 keypoints x (x, y, confidence) = 51 features per frame.
POSE_FEATURE_COLS = [f"{kp}_{axis}" for kp in KEYPOINTS for axis in ("x", "y", "conf")]
N_POSE_FEATURES = len(POSE_FEATURE_COLS)

# Typos present in the hand-authored chapter titles of the source videos.
# Applied wherever raw chapter labels are read, so downstream code only ever
# sees canonical PHASE_ORDER strings.
LABEL_FIXES = {
    "Acceleartion": "Acceleration",
    "Accleration": "Acceleration",
    "Foward Swing Initiation": "Forward Swing Initiation",
    "End of Backstory": "End of Backswing",
}

# Per-frame metadata columns written by the pose-extraction stage, in order,
# ahead of the keypoint columns.
FRAME_META_COLS = [
    "video_id", "video_path", "player",
    "frame_idx", "timestamp_s", "fps",
    "frame_w", "frame_h",
    "video_total_frames", "video_duration_s", "is_slowmo",
    "phase_label", "phase_label_id",
    "chapter_id", "chapter_start_s", "chapter_end_s",
    "chapter_duration_s", "time_since_chapter_start_s",
]

VIDEO_EXTENSIONS = (".mp4", ".mov", ".MP4", ".MOV")


def canonical_label(label: str) -> str:
    """Map a raw chapter title onto its canonical PHASE_ORDER spelling."""
    return LABEL_FIXES.get(label, label)
