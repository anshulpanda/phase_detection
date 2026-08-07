"""
Unit tests for the parts that are easy to break silently.

These run without any data, videos, or checkpoints -- they test invariants of
the maths, not the trained model.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from phase_detection.review_pipeline.confidence import (
    COMPOSITE_MEAN_WEIGHT,
    COMPOSITE_MIN_WEIGHT,
    add_composite_confidence,
    extract_player,
    stratified_sample_by_player,
)
from phase_detection.constants import (
    NUM_CLASSES,
    PHASE_COL_PREFIXES,
    PHASE_ORDER,
    POSE_FEATURE_COLS,
)
from phase_detection.decode import (
    get_phase_boundaries,
    monotonic_decode,
    per_frame_confidence,
    per_phase_confidence,
)
from phase_detection.generate_dataset.features import normalize_pose, resample_sequence
from phase_detection.train.losses import build_ordinal_smoothed_targets, compute_class_weights
from phase_detection.train.model import PhaseTCN, build_model, load_checkpoint, save_checkpoint
from phase_detection.review_pipeline.review import chapters_from_predictions
from phase_detection.train.splits import player_held_out_split, random_video_split


def random_probs(n_frames, n_classes=NUM_CLASSES, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.random((n_frames, n_classes))
    return p / p.sum(axis=1, keepdims=True)


# ── decode ──────────────────────────────────────────────────────────────────
class TestDecode:
    def test_only_legal_transitions_with_reset(self):
        """Each step may stay, advance exactly one, or jump in from no_phase."""
        decoded = monotonic_decode(random_probs(300, seed=1), PHASE_ORDER, allow_no_phase_reset=True)
        no_phase_id = PHASE_ORDER.index("no_phase")
        for prev, cur in zip(decoded[:-1], decoded[1:]):
            assert cur == prev or cur == prev + 1 or prev == no_phase_id, (prev, cur)

    def test_no_reset_variant_is_strictly_monotonic(self):
        """Without the reset edge the path may only stay or advance by one."""
        decoded = monotonic_decode(random_probs(300, seed=2), PHASE_ORDER, allow_no_phase_reset=False)
        assert np.all(np.diff(decoded) >= 0)
        assert np.all(np.diff(decoded) <= 1)

    def test_confident_sequence_is_recovered(self):
        """A clean staircase should decode back to itself."""
        true = np.array([0] * 10 + [1] * 10 + [2] * 10 + [3] * 5 + [4] * 5 + [5] * 5 + [6] * 10)
        probs = np.full((len(true), NUM_CLASSES), 0.01)
        probs[np.arange(len(true)), true] = 0.94
        probs /= probs.sum(axis=1, keepdims=True)
        assert np.array_equal(monotonic_decode(probs, PHASE_ORDER), true)

    def test_boundaries_and_missing_phase(self):
        seq = np.array([0, 0, 1, 1, 1, 6, 6])
        b = get_phase_boundaries(seq, PHASE_ORDER)
        assert b["Start of Unit Turn"] == {"start": 2, "end": 4}
        assert b["Follow Through"] == {"start": 5, "end": 6}
        # A phase that never appears reports None rather than a bogus index.
        assert b["Contact"] == {"start": None, "end": None}

    def test_confidence_follows_decoded_class_not_argmax(self):
        """Confidence must read the decoded class, even when argmax disagrees."""
        probs = np.array([[0.2, 0.8], [0.7, 0.3]])
        decoded = np.array([0, 0])
        np.testing.assert_allclose(per_frame_confidence(probs, decoded), [0.2, 0.7])

    def test_per_phase_confidence_none_when_absent(self):
        probs = random_probs(20, seed=3)
        decoded = np.zeros(20, dtype=int)
        conf = per_phase_confidence(probs, decoded, PHASE_ORDER)
        assert conf["no_phase"] is not None
        assert conf["Contact"] is None


# ── features ────────────────────────────────────────────────────────────────
class TestFeatures:
    def _frame_df(self, n=5, offset=0.0, scale=1.0):
        rng = np.random.default_rng(7)
        data = {c: rng.random(n) * scale + offset for c in POSE_FEATURE_COLS}
        # Give the torso a definite, non-degenerate extent.
        for c in POSE_FEATURE_COLS:
            if c.endswith("_conf"):
                data[c] = np.full(n, 0.9)
        df = pd.DataFrame(data)
        df["left_hip_x"] = 100.0 * scale + offset; df["right_hip_x"] = 120.0 * scale + offset
        df["left_hip_y"] = 200.0 * scale + offset; df["right_hip_y"] = 200.0 * scale + offset
        df["left_shoulder_x"] = 100.0 * scale + offset; df["right_shoulder_x"] = 120.0 * scale + offset
        df["left_shoulder_y"] = 140.0 * scale + offset; df["right_shoulder_y"] = 140.0 * scale + offset
        return df

    def test_translation_invariance(self):
        """Shifting the player in frame must not change normalized features."""
        a = normalize_pose(self._frame_df(offset=0.0))
        b = normalize_pose(self._frame_df(offset=500.0))
        xy = [i for i, c in enumerate(POSE_FEATURE_COLS) if not c.endswith("_conf")]
        np.testing.assert_allclose(a[:, xy], b[:, xy], atol=1e-4)

    def test_confidence_columns_pass_through(self):
        out = normalize_pose(self._frame_df())
        conf_idx = [i for i, c in enumerate(POSE_FEATURE_COLS) if c.endswith("_conf")]
        np.testing.assert_allclose(out[:, conf_idx], 0.9, atol=1e-6)

    def test_degenerate_torso_does_not_divide_by_zero(self):
        df = self._frame_df()
        for c in ["left_shoulder_y", "right_shoulder_y"]:
            df[c] = df["left_hip_y"]
        for c in ["left_shoulder_x", "right_shoulder_x"]:
            df[c] = df["left_hip_x"]
        assert np.all(np.isfinite(normalize_pose(df)))

    def test_resample_changes_length_and_keeps_labels_valid(self):
        feats = np.random.default_rng(0).random((100, 51)).astype(np.float32)
        labels = np.repeat(np.arange(NUM_CLASSES), 100 // NUM_CLASSES + 1)[:100]
        f2, l2 = resample_sequence(feats, labels, 40)
        assert f2.shape == (40, 51) and l2.shape == (40,)
        # Nearest-neighbour must never invent a label that was not present.
        assert set(np.unique(l2)).issubset(set(np.unique(labels)))

    def test_resample_is_noop_at_same_length(self):
        feats = np.zeros((10, 51), dtype=np.float32)
        labels = np.zeros(10, dtype=np.int64)
        f2, l2 = resample_sequence(feats, labels, 10)
        assert f2 is feats and l2 is labels


# ── losses ──────────────────────────────────────────────────────────────────
class TestLosses:
    def test_smoothed_targets_are_distributions(self):
        labels = torch.tensor([[0, 3, 6]])
        t = build_ordinal_smoothed_targets(labels, NUM_CLASSES, 0.1)
        np.testing.assert_allclose(t.sum(-1).numpy(), 1.0, atol=1e-6)

    def test_smoothing_mass_goes_to_ordinal_neighbours(self):
        t = build_ordinal_smoothed_targets(torch.tensor([[3]]), NUM_CLASSES, 0.1)[0, 0]
        assert t[3] == pytest.approx(0.9)
        assert t[2] == pytest.approx(0.05)
        assert t[4] == pytest.approx(0.05)
        assert t[0] == pytest.approx(0.0)

    def test_edge_class_puts_all_smoothing_on_its_single_neighbour(self):
        t = build_ordinal_smoothed_targets(torch.tensor([[0]]), NUM_CLASSES, 0.1)[0, 0]
        assert t[0] == pytest.approx(0.9)
        assert t[1] == pytest.approx(0.1)

    def test_padding_rows_are_all_zero(self):
        t = build_ordinal_smoothed_targets(torch.tensor([[0, -100]]), NUM_CLASSES, 0.1)
        assert t[0, 1].sum() == pytest.approx(0.0)

    def test_class_weights_favour_rare_classes(self):
        sequences = {
            "a": {"labels": np.array([0] * 100 + [5] * 2), "features": None, "player": "P", "fps": 30},
        }
        w = compute_class_weights(sequences, ["a"], NUM_CLASSES)
        assert w[5] > w[0]


# ── model ───────────────────────────────────────────────────────────────────
class TestModel:
    def test_output_shape(self):
        m = build_model()
        out = m(torch.randn(2, 47, 51))
        assert out.shape == (2, 47, NUM_CLASSES)

    def test_receptive_field_matches_known_value(self):
        # 4 blocks, kernel 5, dilations 1/2/4/8 -> 1 + 2*4*(1+2+4+8) = 121
        assert build_model().receptive_field() == 121

    def test_variable_length_is_accepted(self):
        m = build_model()
        for t in (17, 60, 301):
            assert m(torch.randn(1, t, 51)).shape == (1, t, NUM_CLASSES)

    def test_checkpoint_roundtrip(self, tmp_path):
        m = build_model()
        p = tmp_path / "m.pt"
        save_checkpoint(m, p)
        m2 = load_checkpoint(p, "cpu")
        x = torch.randn(1, 33, 51)
        m.eval()
        with torch.no_grad():
            np.testing.assert_allclose(m(x).numpy(), m2(x).numpy(), atol=1e-6)

    def test_loads_bare_state_dict_checkpoints(self, tmp_path):
        """Every checkpoint trained from the original notebooks is a bare state_dict."""
        m = build_model()
        p = tmp_path / "legacy.pt"
        torch.save(m.state_dict(), p)
        assert isinstance(load_checkpoint(p, "cpu"), PhaseTCN)


# ── confidence ──────────────────────────────────────────────────────────────
class TestConfidence:
    def test_extract_player(self):
        assert extract_player("ALCARAZ_FH (10)") == "ALCARAZ"
        assert extract_player("SABALENKA22") == "SABALENKA"

    def test_composite_formula(self):
        row = {f"{p}_confidence": v for p, v in zip(PHASE_COL_PREFIXES, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2])}
        df = add_composite_confidence(pd.DataFrame([row]))
        expected = COMPOSITE_MIN_WEIGHT * 0.2 + COMPOSITE_MEAN_WEIGHT * np.mean([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2])
        assert df["composite_confidence"].iloc[0] == pytest.approx(expected)

    def test_composite_ignores_nan_phases(self):
        vals = [0.9, np.nan, 0.7, 0.6, 0.5, 0.4, 0.3]
        row = {f"{p}_confidence": v for p, v in zip(PHASE_COL_PREFIXES, vals)}
        df = add_composite_confidence(pd.DataFrame([row]))
        assert df["min_confidence"].iloc[0] == pytest.approx(0.3)
        assert df["mean_confidence"].iloc[0] == pytest.approx(np.nanmean(vals))

    def test_composite_weights_the_weakest_phase_hardest(self):
        """One bad phase must drag the composite below the plain mean."""
        vals = [0.95] * 6 + [0.1]
        row = {f"{p}_confidence": v for p, v in zip(PHASE_COL_PREFIXES, vals)}
        df = add_composite_confidence(pd.DataFrame([row]))
        assert df["composite_confidence"].iloc[0] < df["mean_confidence"].iloc[0]

    def test_stratified_sample_hits_target_and_respects_availability(self):
        df = pd.DataFrame({
            "video_id": [f"v{i}" for i in range(50)],
            "player": ["A"] * 30 + ["B"] * 15 + ["C"] * 5,
        })
        rng = np.random.default_rng(42)
        sampled, alloc, _ = stratified_sample_by_player(df, 20, rng)
        assert len(sampled) == 20
        for player, n in alloc.items():
            assert n <= (df["player"] == player).sum()

    def test_stratified_sample_caps_at_pool_size(self):
        df = pd.DataFrame({"video_id": ["a", "b"], "player": ["A", "B"]})
        sampled, _, _ = stratified_sample_by_player(df, 99, np.random.default_rng(0))
        assert len(sampled) == 2


# ── splits ──────────────────────────────────────────────────────────────────
class TestSplits:
    def _sequences(self):
        return {
            f"{p}_{i}": {"features": None, "labels": np.zeros(10), "player": p, "fps": 30}
            for p, n in [("A", 20), ("B", 10), ("C", 5), ("D", 3)]
            for i in range(n)
        }

    def test_player_held_out_has_no_overlap(self):
        seqs = self._sequences()
        train, test = player_held_out_split(seqs, test_player_video_target=8)
        assert set(train) & set(test) == set()
        assert set(train) | set(test) == set(seqs)
        tr = {seqs[v]["player"] for v in train}
        te = {seqs[v]["player"] for v in test}
        assert tr & te == set()

    def test_player_held_out_prefers_long_tail_players(self):
        seqs = self._sequences()
        _, test = player_held_out_split(seqs, test_player_video_target=8)
        # D (3) and C (5) are the smallest; the biggest player must stay in train.
        assert "A" not in {seqs[v]["player"] for v in test}

    def test_random_split_is_deterministic_and_proportioned(self):
        seqs = self._sequences()
        a = random_video_split(seqs, 0.2, seed=1)
        b = random_video_split(seqs, 0.2, seed=1)
        assert a == b
        assert len(a[1]) == pytest.approx(len(seqs) * 0.2, abs=1)


# ── review ──────────────────────────────────────────────────────────────────
class TestReview:
    def _pred_row(self, missing=None):
        row = {"fps": 50.0}
        for phase in PHASE_ORDER:
            prefix = phase.replace(" ", "_")
            if phase == missing:
                row[f"{prefix}_start_frame"] = np.nan
                row[f"{prefix}_end_frame"] = np.nan
            else:
                row[f"{prefix}_start_frame"] = 0
                row[f"{prefix}_end_frame"] = 10
        return pd.Series(row)

    def test_chapters_built_from_complete_predictions(self):
        ch = chapters_from_predictions(self._pred_row())
        assert len(ch) == len(PHASE_ORDER)
        assert [c["label"] for c in ch] == PHASE_ORDER
        # end_s spans through the end of the final frame, not its start instant
        assert ch[0]["end_s"] == pytest.approx(11 / 50.0)

    def test_missing_real_phase_blocks_promotion_to_ground_truth(self):
        assert chapters_from_predictions(self._pred_row(missing="Contact")) is None

    def test_missing_no_phase_is_tolerated(self):
        ch = chapters_from_predictions(self._pred_row(missing="no_phase"))
        assert ch is not None
        assert "no_phase" not in [c["label"] for c in ch]
