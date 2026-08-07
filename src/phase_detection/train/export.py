"""
Model export.

Two targets:

  .pt          TorchScript trace, for any Python/LibTorch consumer.
  .mlpackage   Core ML, for the iOS app.

Both export the network wrapped so it emits **softmax probabilities**, not
logits. The monotonic decode is deliberately *not* baked into the graph -- it
is a dynamic program over the whole sequence that traces poorly, and the client
(the iOS app) applies it to these probabilities itself. Anything consuming the
exported model must implement the decode to reproduce the pipeline's output.
"""

from pathlib import Path

import torch

from ..constants import NUM_CLASSES, PHASE_ORDER
from .model import PhaseTCN, PhaseTCNWithSoftmax

DEFAULT_TRACE_SEQ_LEN = 90  # ~1.5s at 60fps; only needs to be a valid trace shape
DEFAULT_MAX_SEQ_LEN = 600


def _trace(model: PhaseTCN, seq_len: int) -> torch.jit.ScriptModule:
    wrapped = PhaseTCNWithSoftmax(model).eval()
    example_input = torch.randn(1, seq_len, model.n_features)
    return torch.jit.trace(wrapped, example_input)


def export_torchscript(
    model: PhaseTCN,
    output_path: str | Path,
    trace_seq_len: int = DEFAULT_TRACE_SEQ_LEN,
) -> Path:
    """Save a TorchScript trace that outputs per-frame probabilities."""
    model = model.cpu().eval()
    traced = _trace(model, trace_seq_len)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))
    print(f"TorchScript -> {output_path}")
    return output_path


def export_coreml(
    model: PhaseTCN,
    output_path: str | Path,
    trace_seq_len: int = DEFAULT_TRACE_SEQ_LEN,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    minimum_deployment_target: str = "iOS16",
) -> Path:
    """Convert to a Core ML .mlpackage with a dynamic sequence-length axis.

    The time axis is a RangeDim so one model handles clips of any length up to
    `max_seq_len`, rather than needing a fixed frame count.

    Requires the optional `coreml` extra (`pip install -e '.[coreml]'`) and
    only runs on macOS.
    """
    try:
        import coremltools as ct
    except ImportError as e:
        raise ImportError(
            "coremltools is required for Core ML export. "
            "Install it with: pip install -e '.[coreml]'"
        ) from e

    model = model.cpu().eval()
    traced = _trace(model, trace_seq_len)

    seq_len_dim = ct.RangeDim(lower_bound=1, upper_bound=max_seq_len, default=trace_seq_len)
    input_shape = ct.Shape(shape=(1, seq_len_dim, model.n_features))

    target = getattr(ct.target, minimum_deployment_target)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="pose_features", shape=input_shape)],
        outputs=[ct.TensorType(name="phase_probs")],
        minimum_deployment_target=target,
        convert_to="mlprogram",
    )

    mlmodel.short_description = (
        "Tennis forehand swing-phase classifier (TCN over normalized pose keypoints)."
    )
    mlmodel.input_description["pose_features"] = (
        f"Per-frame normalized pose features, shape (1, num_frames, {model.n_features}): "
        "17 COCO keypoints x [x, y, confidence], centered on the hip midpoint and "
        "scaled by torso length."
    )
    mlmodel.output_description["phase_probs"] = (
        f"Per-frame softmax probabilities, shape (1, num_frames, {NUM_CLASSES}), over "
        f"phase classes in order: {PHASE_ORDER}. Apply the monotonic decode client-side."
    )
    mlmodel.user_defined_metadata["phase_order"] = ",".join(PHASE_ORDER)
    mlmodel.user_defined_metadata["n_features"] = str(model.n_features)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    print(f"Core ML -> {output_path}")
    return output_path
