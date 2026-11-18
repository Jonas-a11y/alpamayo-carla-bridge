"""Importable driving lifecycle and CLI for the synchronous CARLA bridge."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .alpamayo_inference import MODEL_IDS, AlpamayoInference, AlpamayoOutput
from .camera_manager import CameraManager
from .carla_client import CarlaClient
from .ego_motion import EgoMotionTracker, EgoState
from .observation import ObservationBuffer
from .trajectory_controller import TrajectoryController, VehicleCommand
from .utils import load_config, setup_logging

logger = logging.getLogger(__name__)


@dataclass
class DriveStep:
    state: EgoState
    output: AlpamayoOutput
    command: VehicleCommand
    frames: dict[str, np.ndarray]


class CarlaBridge:
    """Use as a context manager, then call step() for each model decision.

    Owns its spawned actors and restores the world's previous settings on exit.
    The simulator waits during inference; navigation text can change per step.
    """

    def __init__(self, config: dict[str, Any], recording_dir: str | Path = "recordings") -> None:
        carla_cfg = config.get("carla", {})
        if not carla_cfg.get("sync_mode", True) or carla_cfg.get("fixed_delta", 0.1) != 0.1:
            raise ValueError("The bridge requires synchronous CARLA ticks at fixed_delta=0.1")
        self.config = config
        self.inference = AlpamayoInference(config)
        self.observations = ObservationBuffer(self.inference.variant)
        self._nav_text = config.get("alpamayo", {}).get("nav_text", "")
        if self.inference.variant == "edge" and (not isinstance(self._nav_text, str) or not self._nav_text.strip()):
            raise ValueError("Set alpamayo.nav_text to the intended navigation instruction")
        if self.inference.variant in ("r1", "super") and self._nav_text:
            raise ValueError("Remove alpamayo.nav_text for r1/super; this path does not use navigation")
        self.client = CarlaClient(config)
        self.controller = TrajectoryController(config)
        self._tracker = EgoMotionTracker(origin_x=config.get("vehicle", {}).get("rear_axle_x", 0.0))
        self._cameras = None
        self._recorder = None
        self._recording_dir = recording_dir
        self._started = False
        self._step = 0

    def start(self) -> CarlaBridge:
        if self._started:
            raise RuntimeError("Bridge already started")
        import carla

        try:
            self.observations = ObservationBuffer(self.inference.variant)
            self.controller.reset()
            self._step = 0
            self._recorder = None
            self.inference.load()
            world = self.client.connect()
            vehicle = self.client.spawn_vehicle()
            self._cameras = CameraManager(world, vehicle, self.config)
            self._cameras.spawn_all()
            self.client.attach_collision_sensor()
            vehicle.apply_control(carla.VehicleControl(brake=1.0))
            while not self.observations.ready:
                self._capture()
            if self.config.get("recording", {}).get("enabled", True):
                from .recorder import Recorder

                self._recorder = Recorder(self.config, self._recording_dir)
                self._recorder.initialise(self.client.get_spawn_transform(), world.get_map().name)
            self._started = True
            return self
        except BaseException:
            self.close()
            raise

    def _capture(self) -> tuple[dict, EgoState]:
        frame_id = self.client.tick()
        snapshot = self.client.world.get_snapshot()
        if snapshot.frame != frame_id:
            raise RuntimeError("Another client advanced the world during capture")
        frames = self._cameras.collect_frames(frame_id)
        state = self._tracker.update(self.client.vehicle, snapshot)
        self.observations.append(frames, state)
        if self.client.crashed:
            raise RuntimeError("Collision detected; stopping the bridge")
        return frames, state

    def step(self, nav_text: str | None = None) -> DriveStep:
        if not self._started:
            raise RuntimeError("Call start() or use the bridge as a context manager")
        import carla

        try:
            frames, state = self._capture()
            output = self.inference.predict(self.observations.sample(
                self._nav_text if nav_text is None else nav_text,
            ))
            command = self.controller.compute(output.trajectory, state.speed_kmh)
            if self.client.crashed:
                raise RuntimeError("Collision detected; stopping the bridge")
            self.client.vehicle.apply_control(carla.VehicleControl(**command.to_dict()))
            self._step += 1
            rgb = {name: frame.rgb for name, frame in frames.items()}
            if self._recorder is not None:
                self._recorder.save_step(self._step, rgb, output.trajectory, output.reasoning,
                                         state.to_dict(), command.to_dict())
            return DriveStep(state, output, command, rgb)
        except BaseException:
            self.client.vehicle.apply_control(carla.VehicleControl(brake=1.0))
            raise

    @property
    def session_dir(self) -> Path | None:
        return self._recorder.session_dir if self._recorder is not None else None

    def close(self) -> None:
        try:
            if self._recorder is not None:
                self._recorder.finalise()
        finally:
            try:
                if self._cameras is not None:
                    self._cameras.destroy()
            finally:
                self.client.destroy()
                self._started = False

    def __enter__(self) -> CarlaBridge:
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpamayo–CARLA driving bridge")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML configuration")
    parser.add_argument("--model", choices=tuple(MODEL_IDS), help="Select a variant; switching resets the checkpoint")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--recording-dir", default="recordings")
    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config, model=args.model)
    max_steps = args.max_steps if args.max_steps is not None else config.get("run", {}).get("max_steps", 60)
    if max_steps <= 0:
        parser.error("max-steps must be positive")
    bridge = CarlaBridge(config, args.recording_dir)
    try:
        with bridge:
            for step in range(1, max_steps + 1):
                result = bridge.step()
                logger.info("Step %d/%d | speed=%.1f km/h | steer=%.3f",
                            step, max_steps, result.state.speed_kmh, result.command.steer)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:
        logger.exception("Driving failed")
        raise SystemExit(1)
    if bridge.session_dir is not None and not args.no_video:
        from .video_composer import VideoComposer

        VideoComposer(bridge.session_dir, config).compose()


if __name__ == "__main__":
    main()
