import math

import numpy as np
import pytest
import torch

from bridge import CameraFrame, EgoState, ObservationBuffer, TrajectoryController
from bridge.ego_motion import states_to_ego
from bridge.utils import ego_to_world, world_to_ego


def fill_buffer(variant="edge"):
    buffer = ObservationBuffer(variant)
    for tick in range(21 if variant == "edge" else 16):
        state = EgoState(x=tick, frame=tick, timestamp_us=tick * 100_000)
        frames = {name: CameraFrame(tick, state.timestamp_us,
                  np.full((4, 6, 3), camera * 30 + tick, dtype=np.uint8))
                  for name, camera in buffer.cameras.items()}
        buffer.append(frames, state)
    return buffer


def test_edge_profile_matches_real_upstream_validator_and_prompt():
    from alpamayo2_edge.sample import canonicalize_sample
    from alpamayo2_edge.helper import create_messages
    from types import SimpleNamespace

    sample = fill_buffer().sample("Turn left in 30m")
    data = {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            for key, value in sample.items()}
    canonical = canonicalize_sample(data)
    assert sample["image_frames"].shape == (22, 1, 3, 4, 6)
    assert sample["image_frames"].flags.c_contiguous
    assert sample["camera_indices"].tolist() == [0]*3 + [1]*5 + [2]*3 + [3]*3 + [5]*3 + [6]*5
    assert sample["absolute_timestamps"].tolist() == [
        1_000_000, 1_900_000, 2_000_000,
        0, 1_000_000, 1_500_000, 1_900_000, 2_000_000,
        1_000_000, 1_900_000, 2_000_000,
        1_000_000, 1_900_000, 2_000_000,
        1_000_000, 1_900_000, 2_000_000,
        0, 1_000_000, 1_500_000, 1_900_000, 2_000_000,
    ]
    np.testing.assert_allclose(sample["ego_history_xyz"][0, :, 0], np.arange(-19, 1))
    assert canonical["camera_pixel_bounds"][1] == (163840, 196608)
    assert canonical["camera_pixel_bounds"][0] == (81920, 98304)
    config = SimpleNamespace(history_num_steps=20, tokens_per_history_traj=57,
                             include_camera_ids=True)
    messages = create_messages(data, config)
    content = messages[1]["content"]
    assert sum(item["type"] == "image" for item in content) == 22
    assert any("Turn left in 30m" in item.get("text", "") for item in content)


def test_r1_profile_preserves_camera_and_time_order():
    sample = fill_buffer("r1").sample()
    assert sample["image_frames"].shape == (4, 4, 3, 4, 6)
    assert sample["ego_history_xyz"].shape == (1, 1, 16, 3)
    assert sample["ego_history_rot"].shape == (1, 1, 16, 3, 3)
    np.testing.assert_array_equal(sample["image_frames"][:, :, 0, 0, 0],
                                  [[12, 13, 14, 15], [42, 43, 44, 45],
                                   [72, 73, 74, 75], [192, 193, 194, 195]])


def test_history_warmup_and_invalid_capture_are_explicit_errors():
    buffer = ObservationBuffer()
    with pytest.raises(RuntimeError, match="warming up"):
        buffer.sample("Follow lane")
    with pytest.raises(ValueError, match="Required cameras"):
        buffer.append({}, EgoState())
    full = fill_buffer()
    with pytest.raises(ValueError, match="navigation"):
        full.sample("")
    frames = {name: CameraFrame(21, 2_100_000, np.zeros((4, 6, 3), np.uint8))
              for name in full.cameras}
    frames["front_wide"].frame = 20
    with pytest.raises(ValueError, match="same simulation tick"):
        full.append(frames, EgoState(frame=21, timestamp_us=2_100_000))
    with pytest.raises(ValueError, match="cadence"):
        full.append(frames, EgoState(frame=22, timestamp_us=2_200_000))


def test_yaw_translation_and_left_handed_reflection():
    states = [EgoState(x=10, y=19, yaw=80), EgoState(x=10, y=20, yaw=90)]
    xyz, rotations = states_to_ego(states)
    np.testing.assert_allclose(xyz, [[-1, 0, 0], [0, 0, 0]], atol=1e-6)
    angle = math.radians(10)
    np.testing.assert_allclose(rotations[0], [[math.cos(angle), -math.sin(angle), 0],
                                             [math.sin(angle), math.cos(angle), 0], [0, 0, 1]], atol=1e-6)
    np.testing.assert_allclose(rotations[-1], np.eye(3), atol=1e-6)
    assert ego_to_world(0, 2, 10, 20, 0) == (10, 18)
    assert world_to_ego(10, 18, 10, 20, 0) == (0, 2)
    np.testing.assert_allclose(ego_to_world(2, 3, 10, 20, math.pi/2), [13, 22])


def test_tilted_origin_uses_full_rotation_and_proper_rotations():
    xyz, rotations = states_to_ego([EgoState(x=-1, pitch=30), EgoState(pitch=30)])
    np.testing.assert_allclose(xyz[0], [-math.cos(math.pi/6), 0, 0.5], atol=1e-6)
    np.testing.assert_allclose(rotations, np.broadcast_to(np.eye(3), (2, 3, 3)), atol=1e-6)


def test_controller_steers_left_and_uses_model_time_grid():
    controller = TrajectoryController({"carla": {"fixed_delta": 0.05},
                                       "controller": {"longitudinal_pid": [1, 0, 0]}})
    result = controller.compute([[0.5*i, 0.1*i] for i in range(1, 65)], 10)
    assert result.steer < 0
    assert result.throttle == pytest.approx((math.hypot(0.5, 0.1)*10*3.6 - 10)/30)
    # The first second says stop; distant future motion must not trigger throttle.
    stop = [[0, 0]]*10 + [[i, 0] for i in range(1, 55)]
    result = controller.compute(stop, 10)
    assert result.throttle == 0
    assert result.brake > 0
