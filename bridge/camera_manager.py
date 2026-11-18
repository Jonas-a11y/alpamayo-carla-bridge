"""CARLA camera rig with frame-matched, timestamped RGB capture."""

from __future__ import annotations

import logging
import queue
import time
from typing import Any

import numpy as np

from .observation import CameraFrame, camera_ids

logger = logging.getLogger(__name__)


class CameraManager:
    def __init__(self, world: Any, vehicle: Any, config: dict) -> None:
        self._world = world
        self._vehicle = vehicle
        self._cam_cfg = config["cameras"]
        self._names = camera_ids(config.get("alpamayo", {}).get("variant", "edge"))
        self.resolution = tuple(map(int, self._cam_cfg.get("resolution", [800, 600])))
        self.cameras: dict[str, Any] = {}
        self._queues: dict[str, queue.Queue] = {}

    def spawn_all(self) -> None:
        missing = set(self._names) - self._cam_cfg.keys()
        if missing:
            raise ValueError(f"Missing camera configurations: {sorted(missing)}")
        for name in self._names:
            self._spawn_camera(name, self._cam_cfg[name])

    def collect_frames(self, frame_id: int, timeout: float = 5.0) -> dict[str, CameraFrame]:
        """Discard stale frames and require every camera to match world.tick()."""
        deadline = time.monotonic() + timeout
        frames = {}
        for name in self._names:
            while True:
                try:
                    frame = self._queues[name].get(timeout=max(0, deadline - time.monotonic()))
                except queue.Empty as exc:
                    raise TimeoutError(f"Camera {name} did not deliver frame {frame_id}") from exc
                if frame.frame < frame_id:
                    continue
                if frame.frame > frame_id:
                    raise RuntimeError(f"Camera {name} skipped frame {frame_id}: received {frame.frame}")
                frames[name] = frame
                break
        return frames

    def destroy(self) -> None:
        for actor in reversed(list(self.cameras.values())):
            try:
                actor.stop()
                if actor.is_alive:
                    actor.destroy()
            except Exception as exc:
                logger.debug("Could not destroy camera: %s", exc)
        self.cameras.clear()
        self._queues.clear()

    def _spawn_camera(self, name: str, params: dict) -> None:
        import carla

        bp = self._world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.resolution[0]))
        bp.set_attribute("image_size_y", str(self.resolution[1]))
        bp.set_attribute("fov", str(params.get("fov", 90)))
        bp.set_attribute("sensor_tick", "0.0")
        transform = carla.Transform(
            carla.Location(x=float(params.get("x", 1.5)),
                           y=float(params.get("y", 0)), z=float(params.get("z", 1.6))),
            carla.Rotation(pitch=float(params.get("pitch", 0)), yaw=float(params.get("yaw", 0))),
        )
        sensor = self._world.spawn_actor(bp, transform, attach_to=self._vehicle)
        self.cameras[name] = sensor
        frames: queue.Queue = queue.Queue(maxsize=4)
        self._queues[name] = frames

        def on_image(image: Any) -> None:
            bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
            frame = CameraFrame(image.frame, round(image.timestamp * 1_000_000),
                                bgra[:, :, 2::-1].copy())
            try:
                frames.put_nowait(frame)
            except queue.Full:
                # A preview may tick without collecting. Keep recent frames.
                try:
                    frames.get_nowait()
                except queue.Empty:
                    pass
                frames.put_nowait(frame)

        sensor.listen(on_image)
