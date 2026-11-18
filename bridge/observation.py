"""Timestamped CARLA observations in the native Alpamayo input profiles."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .ego_motion import EgoState, states_to_ego

R1_CAMERAS = {"front_left": 0, "front_wide": 1, "front_right": 2, "front_tele": 6}
EDGE_CAMERAS = {"front_left": 0, "front_wide": 1, "front_right": 2,
                "rear_left": 3, "rear_right": 5, "front_tele": 6}


def camera_ids(variant: str) -> dict[str, int]:
    if variant in ("edge", "super"):
        return EDGE_CAMERAS
    if variant in ("r1", "r15"):
        return R1_CAMERAS
    raise ValueError("variant must be 'r1', 'r15', 'edge' or 'super'")


@dataclass
class CameraFrame:
    frame: int
    timestamp_us: int
    rgb: np.ndarray


class ObservationBuffer:
    """Keep real 10 Hz observations; never pad missing images or ego history."""

    def __init__(self, variant: str = "edge") -> None:
        self.variant = variant
        self.cameras = camera_ids(variant)
        self._observations: deque = deque(maxlen=21 if variant == "edge" else 16)

    def append(self, frames: dict[str, CameraFrame], state: EgoState) -> None:
        if set(frames) != set(self.cameras):
            raise ValueError(f"Required cameras: {list(self.cameras)}")
        if self._observations and abs(state.timestamp_us - self._observations[-1][1].timestamp_us - 100_000) > 1:
            # CARLA accumulates floating-point tick durations; allow microsecond rounding.
            raise ValueError("Observations must have an uninterrupted 0.1-second cadence")
        shapes = set()
        for frame in frames.values():
            if frame.timestamp_us != state.timestamp_us or frame.frame != state.frame:
                raise ValueError("Camera frames and ego pose must belong to the same simulation tick")
            if frame.rgb.dtype != np.uint8 or frame.rgb.ndim != 3 or frame.rgb.shape[2] != 3:
                raise ValueError("Camera images must be uint8 RGB [H, W, 3]")
            shapes.add(frame.rgb.shape)
        if len(shapes) != 1:
            raise ValueError("All camera images must have the same resolution")
        self._observations.append((frames, state))

    @property
    def ready(self) -> bool:
        return len(self._observations) == self._observations.maxlen

    def sample(self, nav_text: str = "") -> dict:
        if not self.ready:
            raise RuntimeError("The camera and ego histories are still warming up")
        if self.variant in ("r1", "super") and nav_text:
            raise ValueError(f"{self.variant} does not support navigation in this trajectory path")
        observations = list(self._observations)
        history_steps = 20 if self.variant == "edge" else 16
        xyz, rot = states_to_ego([state for _, state in observations[-history_steps:]])
        images, ids, timestamps = [], [], []
        for name, camera in self.cameras.items():
            offsets = ((-20, -10, -5, -1, 0) if camera in (1, 6) else (-10, -1, 0))
            if self.variant != "edge":
                offsets = (-3, -2, -1, 0)
            group = [observations[-1 + offset][0][name] for offset in offsets]
            images.append(np.stack([frame.rgb.transpose(2, 0, 1) for frame in group]))
            ids.extend([camera] * len(group))
            timestamps.extend(frame.timestamp_us for frame in group)
        if self.variant == "edge":
            if not isinstance(nav_text, str) or not nav_text.strip():
                raise ValueError("Edge requires a nonempty navigation instruction")
            return {
                "image_frames": np.ascontiguousarray(np.concatenate(images)[:, None]),
                "camera_indices": np.asarray(ids, dtype=np.int64),
                "absolute_timestamps": np.asarray(timestamps, dtype=np.int64),
                "ego_history_xyz": xyz[None], "ego_history_rot": rot[None],
                "nav_text": [nav_text],
            }
        sample = {
            "image_frames": np.ascontiguousarray(np.stack(images)),
            "camera_indices": np.asarray(list(self.cameras.values()), dtype=np.int64),
            "ego_history_xyz": xyz[None, None], "ego_history_rot": rot[None, None],
        }
        if self.variant == "r15":
            sample["nav_text"] = nav_text
        return sample
