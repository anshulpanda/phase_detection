"""
Train/test split strategies.

Three strategies were used across the project's rounds; all are kept because
they answer different questions and their numbers are not interchangeable.

  player_held_out
      Every video from a given player goes entirely to one side. Measures
      generalization to an unseen player -- the honest number for deployment,
      and the default.

  random_video
      Plain random split over videos. Leaks players across the boundary, so it
      reads optimistically; retained because round 3's 80/20 run used it.

  normal_speed_player_held_out
      Player-held-out, but the test pool is restricted to players with zero
      slow-motion videos. The app only ever analyzes normally-captured video,
      so there is no deployment scenario requiring generalization to an unseen
      slow-motion player. Players with any slow-motion footage go entirely to
      train (their clips still contribute signal), leaving a test set with no
      slow-motion/normal-speed confound.

Long-tail (fewest-video) players are preferred for the test side, keeping
high-volume players in train where they do the most good.
"""

import numpy as np
import pandas as pd


def _group_by_player(sequences: dict) -> dict:
    player_videos: dict[str, list] = {}
    for video_id, item in sequences.items():
        player_videos.setdefault(item["player"], []).append(video_id)
    return player_videos


def player_held_out_split(
    sequences: dict,
    test_player_video_target: int = 16,
    seed: int = 42,
) -> tuple[list, list]:
    """Hold out whole players until roughly `test_player_video_target` videos are in test.

    Players are consumed fewest-videos-first, so the test set is made of
    long-tail players and the high-volume players stay in train.
    """
    player_videos = _group_by_player(sequences)
    players_sorted = sorted(player_videos.items(), key=lambda kv: len(kv[1]))

    test_video_ids: list[str] = []
    train_video_ids: list[str] = []
    test_count = 0

    for _player, vids in players_sorted:
        if test_count < test_player_video_target:
            test_video_ids.extend(vids)
            test_count += len(vids)
        else:
            train_video_ids.extend(vids)

    return train_video_ids, test_video_ids


def random_video_split(
    sequences: dict,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list, list]:
    """Random split over videos, ignoring player identity.

    Note this leaks players across the split -- the same player appears in both
    train and test -- so the resulting score is optimistic relative to
    `player_held_out_split`.
    """
    rng = np.random.default_rng(seed)
    video_ids = sorted(sequences.keys())
    shuffled = list(rng.permutation(video_ids))
    n_test = int(round(len(shuffled) * test_fraction))
    return shuffled[n_test:], shuffled[:n_test]


def normal_speed_player_held_out_split(
    sequences: dict,
    slowmo_map: pd.Series,
    test_player_video_target: int = 150,
    verbose: bool = True,
) -> tuple[list, list]:
    """Player-held-out with a zero-slow-motion test pool.

    Any player with at least one slow-motion video is forced entirely into
    train; the held-out test set is drawn only from players with none.
    """
    player_videos = _group_by_player(sequences)
    player_has_slowmo: dict[str, bool] = {}
    for video_id, item in sequences.items():
        if slowmo_map.get(video_id, False):
            player_has_slowmo[item["player"]] = True

    eligible = {p: v for p, v in player_videos.items() if not player_has_slowmo.get(p, False)}
    forced_train = {p: v for p, v in player_videos.items() if player_has_slowmo.get(p, False)}

    if verbose:
        print(
            f"Players with >=1 slow motion video (forced to train): {len(forced_train)} players, "
            f"{sum(len(v) for v in forced_train.values())} videos"
        )
        print(
            f"Players with zero slow motion videos (test-eligible): {len(eligible)} players, "
            f"{sum(len(v) for v in eligible.values())} videos"
        )

    train_video_ids: list[str] = []
    for vids in forced_train.values():
        train_video_ids.extend(vids)

    test_video_ids: list[str] = []
    test_count = 0
    for _player, vids in sorted(eligible.items(), key=lambda kv: len(kv[1])):
        if test_count < test_player_video_target:
            test_video_ids.extend(vids)
            test_count += len(vids)
        else:
            train_video_ids.extend(vids)

    return train_video_ids, test_video_ids


SPLIT_STRATEGIES = {
    "player_held_out": player_held_out_split,
    "random_video": random_video_split,
    "normal_speed_player_held_out": normal_speed_player_held_out_split,
}


def summarize_split(sequences: dict, train_ids: list, test_ids: list) -> None:
    """Print split composition and assert no player straddles the boundary."""
    train_players = sorted({sequences[v]["player"] for v in train_ids})
    test_players = sorted({sequences[v]["player"] for v in test_ids})
    overlap = set(train_players) & set(test_players)

    train_frames = sum(len(sequences[v]["labels"]) for v in train_ids)
    test_frames = sum(len(sequences[v]["labels"]) for v in test_ids)

    print(f"Train: {len(train_ids)} videos, {train_frames} frames, players: {train_players}")
    print(f"Test:  {len(test_ids)} videos, {test_frames} frames, players: {test_players}")
    if overlap:
        print(f"Player overlap between train/test: {sorted(overlap)}  <-- expected for random_video")
    else:
        print("Player overlap between train/test: none")
