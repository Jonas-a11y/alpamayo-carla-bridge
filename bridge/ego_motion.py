"""Ego-vehicle motion state tracker."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class EgoState:
    """Snapshot of the ego vehicle's kinematic state at one time step.

    All angles are in **degrees** (matching CARLA conventions) unless noted.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    speed_kmh: float = 0.0
    heading: float = 0.0  # yaw in degrees
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    frame: int = 0
    timestamp_us: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Serialise to a plain dict (for JSONL recording)."""
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "z": round(self.z, 4),
            "speed_kmh": round(self.speed_kmh, 2),
            "heading": round(self.heading, 4),
            "yaw": round(self.yaw, 4),
            "pitch": round(self.pitch, 4),
            "roll": round(self.roll, 4),
            "frame": self.frame,
            "timestamp_us": self.timestamp_us,
        }


@dataclass
class EgoMotionTracker:
    """Read CARLA poses at the configured longitudinal ego origin."""

    origin_x: float = 0.0

    def update(self, vehicle: Any, snapshot: Any) -> EgoState:
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        speed_ms = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

        pitch = math.radians(transform.rotation.pitch)
        yaw = math.radians(transform.rotation.yaw)
        return EgoState(
            x=transform.location.x + self.origin_x * math.cos(pitch) * math.cos(yaw),
            y=transform.location.y + self.origin_x * math.cos(pitch) * math.sin(yaw),
            z=transform.location.z + self.origin_x * math.sin(pitch),
            speed_kmh=speed_ms * 3.6,
            heading=transform.rotation.yaw,
            yaw=transform.rotation.yaw,
            pitch=transform.rotation.pitch,
            roll=transform.rotation.roll,
            frame=snapshot.frame,
            timestamp_us=round(snapshot.timestamp.elapsed_seconds * 1_000_000),
        )


def states_to_ego(states: list[EgoState]) -> tuple[np.ndarray, np.ndarray]:
    """Reflect CARLA's left-handed poses, then express them in the current ego frame.

    CARLA local Y points right; Alpamayo Y points left. Reflect both the world
    and body bases so rotations remain proper SO(3) matrices, including tilt.
    """
    xyz = np.array([[s.x, -s.y, s.z] for s in states], dtype=np.float64)
    rotations = []
    reflection = np.diag([1.0, -1.0, 1.0])
    for state in states:
        pitch, yaw, roll = map(math.radians, (state.pitch, state.yaw, state.roll))
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cr, sr = math.cos(roll), math.sin(roll)
        carla_rot = np.array([
            [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr],
        ])
        rotations.append(reflection @ carla_rot @ reflection)
    rotations = np.stack(rotations)
    if not np.isfinite(xyz).all() or not np.isfinite(rotations).all():
        raise ValueError("Ego poses must be finite")
    inverse = rotations[-1].T
    return ((xyz - xyz[-1]) @ inverse.T).astype(np.float32), (inverse @ rotations).astype(np.float32)
