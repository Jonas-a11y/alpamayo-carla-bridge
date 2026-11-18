import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bridge import AlpamayoOutput, CarlaBridge, load_config
from bridge.camera_manager import CameraManager


class Actor:
    def __init__(self, kind, world):
        self.kind = kind
        self.world = world
        self.is_alive = True
        self.id = len(world.actors)
        self.listener = None
        self.controls = []

    def listen(self, callback):
        self.listener = callback

    def stop(self):
        self.listener = None

    def destroy(self):
        self.is_alive = False

    def get_transform(self):
        return SimpleNamespace(location=SimpleNamespace(x=10, y=20, z=1),
                               rotation=SimpleNamespace(yaw=90, pitch=0, roll=0))

    def get_velocity(self):
        return SimpleNamespace(x=0, y=0, z=0)

    def apply_control(self, control):
        self.controls.append(control)

    def emit(self, frame, timestamp=None):
        self.listener(SimpleNamespace(frame=frame, timestamp=frame/10 if timestamp is None else timestamp,
                                      height=4, width=6, raw_data=bytes([1, 2, 3, 255])*24))


class World:
    def __init__(self):
        self.frame = 0
        self.actors = []
        self.settings = SimpleNamespace(synchronous_mode=False, fixed_delta_seconds=None)

    def get_settings(self):
        return copy.copy(self.settings)

    def apply_settings(self, settings):
        self.settings = settings

    def get_map(self):
        return SimpleNamespace(name="TestTown", get_spawn_points=lambda: [
            SimpleNamespace(location=SimpleNamespace(x=0, y=0, z=0))])

    def get_blueprint_library(self):
        def find(kind):
            return SimpleNamespace(kind=kind, set_attribute=lambda key, value: None)
        return SimpleNamespace(find=find)

    def spawn_actor(self, blueprint, transform, **kwargs):
        actor = Actor(blueprint.kind, self)
        self.actors.append(actor)
        return actor

    def tick(self):
        self.frame += 1
        for actor in self.actors:
            if actor.kind == "sensor.camera.rgb" and actor.listener is not None:
                actor.emit(self.frame)
        return self.frame

    def get_snapshot(self):
        return SimpleNamespace(frame=self.frame, timestamp=SimpleNamespace(elapsed_seconds=self.frame/10))


@pytest.fixture
def simulation(monkeypatch):
    world = World()
    server = SimpleNamespace(set_timeout=lambda value: None, get_available_maps=lambda: [],
                             get_world=lambda: world)
    carla = SimpleNamespace(Client=lambda host, port: server,
        Location=lambda **kwargs: SimpleNamespace(**kwargs),
        Rotation=lambda **kwargs: SimpleNamespace(**kwargs),
        Transform=lambda *args: args,
        VehicleControl=lambda throttle=0, steer=0, brake=0: SimpleNamespace(
            throttle=throttle, steer=steer, brake=brake))
    monkeypatch.setitem(sys.modules, "carla", carla)
    config = load_config()
    config["recording"]["enabled"] = False
    return config, world


def test_capture_discards_stale_frames_and_reports_missing_or_future_frames(simulation):
    config, world = simulation
    manager = CameraManager(world, object(), config)
    manager.spawn_all()
    for sensor in manager.cameras.values():
        sensor.emit(1)
        sensor.emit(2)
    frames = manager.collect_frames(2)
    assert all(frame.frame == 2 and frame.timestamp_us == 200_000 for frame in frames.values())
    np.testing.assert_array_equal(frames["front_wide"].rgb[0, 0], [3, 2, 1])
    with pytest.raises(TimeoutError, match="did not deliver"):
        manager.collect_frames(3, timeout=0)
    for sensor in manager.cameras.values():
        sensor.emit(4)
    with pytest.raises(RuntimeError, match="skipped frame 3"):
        manager.collect_frames(3)
    manager.destroy()
    assert all(not actor.is_alive for actor in world.actors)


def test_bridge_warmup_step_failure_and_cleanup(simulation, monkeypatch):
    config, world = simulation
    bridge = CarlaBridge(config)
    monkeypatch.setattr(bridge.inference, "load", lambda: None)

    def predict(sample):
        assert sample["image_frames"].shape == (22, 1, 3, 4, 6)
        assert sample["nav_text"] == ["Turn right in 30m"]
        return AlpamayoOutput([[i/10, 0] for i in range(1, 65)], "reasoning")

    monkeypatch.setattr(bridge.inference, "predict", predict)
    with bridge:
        assert world.frame == 21
        assert world.settings.synchronous_mode
        result = bridge.step("Turn right in 30m")
        assert result.state.frame == 22
        assert result.state.timestamp_us == 2_200_000
        np.testing.assert_allclose([result.state.x, result.state.y], [10, 18.6])
        assert result.command.throttle > 0
        assert len(result.frames) == 6
        vehicle = bridge.client.vehicle
        monkeypatch.setattr(bridge.inference, "predict", lambda sample: (_ for _ in ()).throw(RuntimeError("model failed")))
        with pytest.raises(RuntimeError, match="model failed"):
            bridge.step()
        assert vehicle.controls[-1].throttle == 0
        assert vehicle.controls[-1].brake == 1
    assert all(not actor.is_alive for actor in world.actors)
    assert world.settings.synchronous_mode is False
    assert world.settings.fixed_delta_seconds is None


def test_start_failure_also_destroys_partially_spawned_actors(simulation, monkeypatch):
    config, world = simulation
    del config["cameras"]["rear_left"]
    bridge = CarlaBridge(config)
    monkeypatch.setattr(bridge.inference, "load", lambda: None)
    with pytest.raises(ValueError, match="Missing camera"):
        bridge.start()
    assert all(not actor.is_alive for actor in world.actors)
    assert world.settings.synchronous_mode is False


def test_bridge_rejects_incompatible_simulation_cadence(simulation):
    config, _ = simulation
    config["carla"]["fixed_delta"] = 0.05
    with pytest.raises(ValueError, match="fixed_delta"):
        CarlaBridge(config)


def test_recording_finalization_error_still_restores_world_and_destroys_actors(simulation, monkeypatch, tmp_path):
    from bridge.recorder import Recorder

    config, world = simulation
    config["recording"]["enabled"] = True
    bridge = CarlaBridge(config, recording_dir=tmp_path)
    monkeypatch.setattr(bridge.inference, "load", lambda: None)
    original_finalise = Recorder.finalise

    def fail_finalise(recorder):
        original_finalise(recorder)
        raise OSError("recording flush failed")

    monkeypatch.setattr(Recorder, "finalise", fail_finalise)
    with pytest.raises(OSError, match="recording flush failed"):
        with bridge:
            pass
    assert all(not actor.is_alive for actor in world.actors)
    assert world.settings.synchronous_mode is False
    assert world.settings.fixed_delta_seconds is None
