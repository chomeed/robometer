#!/usr/bin/env python3
"""Load the board-insertion LeRobot splits for Robometer/RFM conversion.

The loader intentionally reads the LeRobot v3 files directly (Parquet metadata,
Parquet actions, and sharded MP4 video). This keeps it usable in Robometer's
environment without requiring the ``lerobot`` package to be installed.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd


CAMERA_KEY = "observation.images.right_wrist"
TASK_NAME = "insert the board into the slot"
VALID_SPLITS = {"train", "test"}


def _decode_video_segment(video_path: str, start_frame: int, end_frame: int) -> list[np.ndarray]:
    """Decode an exclusive frame range from a sharded LeRobot video."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    frames: list[np.ndarray] = []
    try:
        # LeRobot videos are keyframed frequently. Seeking avoids decoding every
        # earlier episode in a shard; the frame-index checks retain exact bounds.
        seek_timestamp = max(0, int((start_frame / float(stream.average_rate)) / stream.time_base))
        container.seek(seek_timestamp, stream=stream, backward=True)
        for frame in container.decode(stream):
            frame_index = round(float(frame.time) * float(stream.average_rate))
            if frame_index < start_frame:
                continue
            if frame_index >= end_frame:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
    finally:
        container.close()

    expected = end_frame - start_frame
    if len(frames) != expected:
        raise RuntimeError(
            f"Decoded {len(frames)} frames from {video_path}, expected {expected} "
            f"for [{start_frame}, {end_frame})"
        )
    return frames


def _read_episode_metadata(dataset_root: Path) -> pd.DataFrame:
    files = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata found under {dataset_root}")
    return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True).sort_values("episode_index")


def _read_data(dataset_root: Path) -> pd.DataFrame:
    files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data parquet files found under {dataset_root}")
    return pd.concat((pd.read_parquet(path, columns=["episode_index", "action"]) for path in files), ignore_index=True)


def _load_source(dataset_root: Path, quality_label: str, split: str) -> list[dict[str, Any]]:
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    fps = int(info["fps"])
    video_template = info["video_path"]
    episodes = _read_episode_metadata(dataset_root)
    data = _read_data(dataset_root)
    actions_by_episode = {
        int(episode_index): np.asarray(group["action"].tolist(), dtype=np.float32)
        for episode_index, group in data.groupby("episode_index", sort=False)
    }

    trajectories: list[dict[str, Any]] = []
    for _, row in episodes.iterrows():
        episode_index = int(row["episode_index"])
        actions = actions_by_episode[episode_index]

        chunk_index = int(row[f"videos/{CAMERA_KEY}/chunk_index"])
        file_index = int(row[f"videos/{CAMERA_KEY}/file_index"])
        start_frame = round(float(row[f"videos/{CAMERA_KEY}/from_timestamp"]) * fps)
        end_frame = start_frame + int(row["length"])
        video_path = dataset_root / video_template.format(
            video_key=CAMERA_KEY,
            chunk_index=chunk_index,
            file_index=file_index,
        )
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        if len(actions) != int(row["length"]):
            raise ValueError(
                f"{dataset_root.name} episode {episode_index}: "
                f"{len(actions)} actions != {row['length']} frames"
            )

        trajectories.append(
            {
                "id": f"board_insertion_{split}_{quality_label}_{episode_index:06d}",
                "frames": partial(
                    _decode_video_segment,
                    str(video_path),
                    start_frame,
                    end_frame,
                ),
                "actions": actions,
                "is_robot": True,
                "task": TASK_NAME,
                "optimal": "optimal" if quality_label == "successful" else "failed",
                "quality_label": quality_label,
                "partial_success": 1.0 if quality_label == "successful" else 0.0,
                "data_source": f"board_insertion_{split}_rfm",
                "preference_group_id": None,
                "preference_rank": None,
            }
        )
    return trajectories


def load_board_insertion_rfm_dataset(base_path: str, split: str) -> dict[str, list[dict[str, Any]]]:
    """Merge a success/failure split into Robometer loader trajectories.

    Args:
        base_path: Directory containing ``success_train``, ``failure_train``,
            ``success_test``, and ``failure_test`` LeRobot datasets.
        split: Either ``"train"`` or ``"test"``.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {split!r}")

    base = Path(base_path).expanduser().resolve()
    success_root = base / f"success_{split}"
    failure_root = base / f"failure_{split}"
    for root in (success_root, failure_root):
        if not (root / "meta" / "info.json").exists():
            raise FileNotFoundError(f"LeRobot dataset not found: {root}")

    successful = _load_source(success_root, "successful", split)
    failed = _load_source(failure_root, "failure", split)
    # Proportionally interleave both labels. Besides avoiding class-sized blocks,
    # this ensures --output.max_trajectories=N samples across the whole split
    # instead of trimming all failures from a capped conversion.
    ranked = [
        ((index + 0.5) / len(group), label_order, item)
        for label_order, group in enumerate((successful, failed))
        for index, item in enumerate(group)
    ]
    merged = [item for _, _, item in sorted(ranked, key=lambda value: (value[0], value[1]))]
    print(
        f"Loaded board-insertion {split} split: {len(successful)} successful + "
        f"{len(failed)} failure = {len(merged)} trajectories"
    )
    return {"board_insertion": merged}
