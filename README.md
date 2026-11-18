# Alpamayo–CARLA Bridge

An importable Python bridge between CARLA and NVIDIA's Alpamayo action experts.
It supports **Alpamayo 2 Edge** (the default) and **Alpamayo 1 / R1**, returning
64 ego-frame waypoints at +0.1 through +6.4 seconds and their reasoning.
The bridge uses native trajectory APIs and propagates failures instead of
substituting fabricated trajectories.

## Install the core on any laptop

From this checkout:

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
alpamayo-carla-drive --help
~~~

The distribution is named 'alpamayo-carla-bridge'; the import is 'bridge'.
The core needs only NumPy and PyYAML. Importing it does not require CARLA,
PyTorch, Transformers, model weights, or a network connection. Model and video
packages are installed separately.

## Edge runtime on a GPU host

Use **Python 3.12**, a Linux host with a compatible NVIDIA CUDA driver and BF16
support, and matching CARLA server/client versions. NVIDIA recommends at least
16 GB VRAM for Edge BF16 inference; more candidates can require more memory.
The Mac can install the core and run CPU checks, but full model inference and
a CARLA server on macOS have not been validated here.

~~~bash
pip install -e '.[edge,carla,video]'
hf auth login
# Start a CARLA 0.9.16 server in another terminal, then:
alpamayo-carla-drive --config config.yaml --max-steps 60
~~~

The Edge extra pins upstream code and its native dependencies (PyTorch 2.8.0,
torchvision 0.23.0, Transformers 5.17.0). SDPA is the default; FlashAttention is
optional. The config pins the released Edge checkpoint revision. Obtain model
access on [Hugging Face](https://huggingface.co/nvidia/Alpamayo2-Edge).
CARLA observations do not require the PhysicalAI-AV dataset or its SDK.

The CARLA extra installs 0.9.16, which has Python 3.12 wheels for Linux x86_64
and Windows. Install a client wheel matching your actual server. CARLA 0.9.15's
published wheels do not cover Python 3.12; it requires a compatible custom build
if you must retain that server. Windows uses the same package commands after
activating the venv with '.venv/Scripts/Activate.ps1' in PowerShell; NVIDIA's
Edge inference reference runtime is Linux.

Optional MP4 composition requires ffmpeg on PATH. Use '--no-video' to skip it,
and 'recording.enabled: false' to disable frame recording.

## Python API

~~~python
from bridge import CarlaBridge, load_config

config = load_config("/path/to/alpamayo-carla-bridge/config.yaml")
config["recording"]["enabled"] = False

with CarlaBridge(config) as bridge:
    for _ in range(60):
        result = bridge.step(nav_text="Follow the current lane.")
        print(result.state.timestamp_us, result.output.trajectory, result.output.reasoning)
~~~

'start()' loads the model, connects, spawns sensors and captures a real history
while braking. The context manager destroys its actors and restores previous
world settings. Each 'step()' advances a tick, matches camera frame IDs to that
tick, prepares the native input, runs inference and applies control. The simulator
waits while the model computes. An inference error, missing sensor, incorrect
frame or collision raises an error; a failed step commands a brake. CLI failures
exit nonzero.

For inference on already prepared observations:

~~~python
from bridge import AlpamayoInference, ObservationBuffer

inference = AlpamayoInference(config)
inference.load()
buffer = ObservationBuffer("edge")
# At each 0.1-second tick: buffer.append(camera_frames, ego_state)
# camera_frames: dict[str, CameraFrame(frame, timestamp_us, rgb_uint8)]
# ego_state: EgoState with the same frame/timestamp, pose and CARLA angles.
# After buffer.ready:
output = inference.predict(buffer.sample("Turn right in 30m"))
~~~

'predict()' also accepts the raw tensor dictionary documented by the matching
upstream loader. It uses candidate zero and its reasoning, without ranking Edge's
placeholder log probabilities or using future ground truth.

## Input profiles and coordinates

| Variant | Cameras | Image schedule relative to the current tick | Ego history |
| --- | --- | --- | --- |
| Edge | IDs 0, 1, 2, 3, 5, 6 | Front wide/tele: −2, −1, −0.5, −0.1, 0 s; others: −1, −0.1, 0 s (22 images) | 20 poses, −1.9 through 0 s |
| R1 | IDs 0, 1, 2, 6 | Four frames per camera: −0.3, −0.2, −0.1, 0 s | 16 poses, −1.5 through 0 s |

The bridge requires synchronous 0.1-second ticks and never pads missing images
or poses. Images retain their captured simulation timestamps. Edge's helper
applies checkpoint-native normalization, camera/time labels and pixel budgets.

Model XY uses **x forward, y left, meters**. CARLA uses Y right. The bridge
reflects position and rotation bases and transforms history into the current ego
frame, including pitch and roll. The controller steers left for positive model Y
and estimates speed using the model's fixed 0.1-second waypoint spacing.

Edge's training origin is the rear axle. 'vehicle.rear_axle_x: -1.4' is an
approximate Model 3 offset from the CARLA actor origin; calibrate it for the actual
vehicle. State and recorded positions refer to this origin. Camera attachment
coordinates refer to the actor origin. The rig approximates the training rig;
neither its calibration nor driving quality has been measured in CARLA here.
See [camera configuration](docs/CAMERA_CONFIG.md).

'alpamayo.nav_text' is explicit driving intent. The default requests lane following;
it is not a route generated from CARLA's map. Change it per step or in the config.

## Alpamayo 1 / R1

Use a **separate Python 3.12 environment**: R1's Transformers 4.57.1 conflicts
with Edge's Transformers 5.17.0. The R1 extra provides SDPA runtime dependencies;
install upstream code without its mandatory FlashAttention dependency, which
is unused by the selected SDPA path:

~~~bash
pip install -e '.[r1,carla,video]'
pip install --no-deps 'alpamayo_r1 @ git+https://github.com/NVlabs/alpamayo.git@11a0e01c13a5622377c45ee37d653351453ec43b'
hf auth login
~~~

In a copy of 'config.yaml' set 'alpamayo.variant: r1' and
'alpamayo.model_id: nvidia/Alpamayo-R1-10B', and remove the Edge 'revision'.
R1 uses four cameras and the native
'sample_trajectories_from_data_with_vlm_rollout' method. NVIDIA recommends at
least 24 GB VRAM for R1. CPU float32 is configurable for experimentation but
full CPU inference is not validated by these checks.

## Verification

~~~bash
pip install -e '.[edge,test]'
python -m pytest -q
~~~

CPU checks exercise the real Edge sample validator and prompt builder, native
adapter contracts using mocked model outputs, coordinate conversion, camera
frame matching, braking on inference failure and lifecycle cleanup. CI runs
without model weights or a CARLA server. Passing establishes API and data-contract
behavior, not CUDA inference quality or closed-loop performance. A GPU/CARLA
smoke run remains necessary on the target host.

Source-checkout scripts remain available:

~~~bash
python scripts/run_drive.py --config config.yaml
python scripts/setup_carla_cameras.py --config config.yaml
python scripts/compose_video.py recordings/SESSION --config config.yaml
~~~

Recordings contain every active camera and JSONL telemetry. The composite video
currently displays the four front views plus trajectories and reasoning.

## References

- [Alpamayo 2 Edge code and input contract](https://github.com/NVlabs/alpamayo2-edge)
- [Alpamayo 2 Edge checkpoint](https://huggingface.co/nvidia/Alpamayo2-Edge)
- [Alpamayo 1 / R1 code](https://github.com/NVlabs/alpamayo)
- [CARLA documentation](https://carla.readthedocs.io/)
- [Architecture](docs/ARCHITECTURE.md)

MIT project; dependencies and model weights have their own licenses.
