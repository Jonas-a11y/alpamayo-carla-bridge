# Tutorials

[README](../README.md) · [Model setup](MODELS.md) · [Troubleshooting](TROUBLESHOOTING.md)

## 1. Install and import on a laptop

This checks the Python package without a simulator, CUDA or model downloads.
Run the following from a terminal with Git and Python 3.10+ installed:

~~~bash
git clone https://github.com/Jonas-a11y/alpamayo-carla-bridge.git
cd alpamayo-carla-bridge
python -m venv .venv
source .venv/bin/activate
pip install -e .
alpamayo-carla-drive --help
python -c 'from bridge import CarlaBridge, AlpamayoInference, ObservationBuffer; print("Import OK")'
~~~

On Windows use `.venv/Scripts/Activate.ps1` to activate in PowerShell.
Expected results: the CLI lists `--model {r1,r15,edge,super}` and the final
command prints `Import OK`. Importing creates no vehicle and loads no weights.
Starting a drive requires the target-host setup below. macOS can run core/CPU
checks, but the published CARLA 0.9.16 Python 3.12 wheels target Linux x86_64
and Windows; CUDA inference requires an NVIDIA GPU host.

## 2. First Edge drive

Use a dedicated CARLA 0.9.16 server and a Python 3.12 environment on the target
GPU host. The bridge loads the configured map if available and spawns a vehicle
at a random spawn point. Only one client should advance this synchronous world.

1. Clone the repository as above, then install the
   [Edge runtime](MODELS.md#edge) in its own environment. Check CUDA availability:

   ~~~bash
   python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
   ~~~

   Expect `True` and your GPU's name. Log in with `hf auth login` and complete
   any access/license steps on the checkpoint's Hugging Face page. The first
   drive downloads uncached weights and processor assets.

2. Start the server in another terminal. For a Linux CARLA archive, run this
   from its extracted directory (the default RPC port is 2000):

   ~~~bash
   ./CarlaUE4.sh -RenderOffScreen
   ~~~

   On Windows, start `CarlaUE4.exe`. Offscreen rendering still produces camera
   images; CARLA's no-rendering mode does not provide the RGB input needed here.
   Keep the server running while using the bridge.

3. In the bridge checkout, copy `config.yaml` to `my-config.yaml` and edit it.
   Set `carla.host`/`port` to your server, choose an installed `carla.world`,
   and set `alpamayo.nav_text` to your intended instruction, such as
   `Follow the current lane.` Keep `sync_mode: true` and `fixed_delta: 0.1`.
   Navigation is supplied text; the bridge has no map-based route planner.

4. Inspect the cameras before loading the model:

   ~~~bash
   python scripts/setup_carla_cameras.py --config my-config.yaml
   ~~~

   Open the images under `camera_preview/`. Check visibility, orientation and
   occlusion, then adjust the [camera rig and rear-axle offset](CAMERA_CONFIG.md).
   Preview does not use model weights; it needs CARLA and Pillow from the video extra.

5. Run a short session:

   ~~~bash
   alpamayo-carla-drive --config my-config.yaml --model edge --max-steps 10 --no-video
   ~~~

   The bridge loads the model, brakes during 21 warmup ticks, then logs ten
   model decisions. Ten decisions advance one second of simulation, in addition
   to warmup; elapsed wall time depends on inference and rendering. During
   inference the synchronous simulator waits. Successful exit destroys spawned
   actors and restores the world's previous tick settings.

`--no-video` skips MP4 composition, but still records by default. To disable
both recording and composition, set `recording.enabled: false` in the YAML.
An inference/capture/collision error stops the run with a nonzero CLI exit status;
inspect the traceback using the [troubleshooting guide](TROUBLESHOOTING.md).

## 3. Use the bridge in your own Python project

Install this repository into your application's environment, alongside the
appropriate [model runtime](MODELS.md) and CARLA client. For example, with an
existing local checkout:

~~~bash
pip install -e /path/to/alpamayo-carla-bridge
~~~

The YAML and tutorial scripts are checkout files, not bundled package resources.
Pass an explicit config path when running outside the checkout. Save this as
`drive.py`, replacing the path with your own:

~~~python
from bridge import CarlaBridge, load_config

config = load_config("/path/to/alpamayo-carla-bridge/config.yaml", model="edge")
config["recording"]["enabled"] = False

with CarlaBridge(config) as bridge:
    for _ in range(10):
        result = bridge.step(nav_text="Follow the current lane.")
        print("simulation microseconds:", result.state.timestamp_us)
        print("first waypoint:", result.output.trajectory[0])
        print("reasoning:", result.output.reasoning)
        print("control:", result.command.to_dict())
~~~

Run with `python drive.py`. The context manager handles start and cleanup even
when the loop raises. `step()` captures one tick, predicts, applies control and
returns a `DriveStep`:

| Field | Meaning |
| --- | --- |
| `state` | Rear-axle-origin CARLA pose, speed in km/h, frame ID and simulation timestamp |
| `output.trajectory` | 64 `[x, y]` waypoints in meters, X forward/Y left, relative to the current ego pose |
| `output.reasoning` | The selected candidate's reasoning text |
| `command` | Applied throttle, CARLA steer and brake |
| `frames` | Current camera RGB arrays, `uint8 [H, W, 3]` |

Navigation passed to `step()` applies to that decision. Omit it to use the YAML
instruction. After closing, the same `CarlaBridge` can start a new session with
fresh history and reuse its loaded model. To change model/device/checkpoint,
create a new bridge with the desired configuration.

### Switch to Alpamayo 1.5, R1 or Super

Use the separate Nano/Super environment and install the relevant upstream code
as described in [model setup](MODELS.md#alpamayo-1-15-and-2-super). For 1.5:

~~~python
config = load_config("/path/to/config.yaml", model="r15")
config["recording"]["enabled"] = False
with CarlaBridge(config) as bridge:
    result = bridge.step(nav_text="Turn right in 30m")
~~~

For R1 use `model="r1"` and `bridge.step()`; for Super use `model="super"`
and `bridge.step()`. Those two paths reject navigation text. 1.5 accepts an
optional instruction. Selecting another variant clears the old checkpoint
ID/revision; set custom checkpoint overrides after selecting the variant.

## 4. Record and inspect a run

With `recording.enabled: true`, choose your output root:

~~~bash
alpamayo-carla-drive --config my-config.yaml --max-steps 10 --recording-dir recordings --no-video
~~~

Each session has `frames/step_0001_<camera>.jpg`, `data.jsonl` and
`metadata.json`. The JSONL contains predicted paths, reasoning, applied control
and actual poses; `ego_state.timestamp_us` is simulation time, while the top-level
`timestamp` is recording wall time. Metadata contains the configuration and spawn
information. Recordings store current views, not a complete replayable input
history or a CARLA world-state snapshot.

Install the video extra and ffmpeg to render later, without CUDA or CARLA:

~~~bash
pip install -e '.[video]'
ffmpeg -version
python scripts/compose_video.py recordings/YOUR_SESSION --config my-config.yaml
~~~

Replace `YOUR_SESSION` with the created directory name. The output is
`recordings/YOUR_SESSION/output.mp4`, showing the four front views, reasoning
and predicted/actual paths. Edge and Super recordings also retain rear-camera
JPEGs. Omitting `--no-video` composes the video automatically after a successful
CLI run. In Python, the session path is available as `bridge.session_dir`.

## 5. Run checks without weights

Follow the two commands under [CPU contract checks](MODELS.md#cpu-contract-checks)
in their respective environments. They check the upstream input/prompt adapters
and simulated CARLA lifecycle without downloading checkpoints. A passing suite
does not validate driving quality. Evaluate camera calibration, inference
memory/latency and closed-loop behavior on the target CUDA/CARLA host.

## 6. Validate on Google Colab

Download [alpamayo_colab.ipynb](../notebooks/alpamayo_colab.ipynb), upload it to
[Google Colab](https://colab.research.google.com/), choose a GPU runtime, and run
the cells in order. The notebook installs managed Python 3.12 environments for
Edge and Nano/Super, runs both CPU contract suites and verifies a built core wheel.
It does not change the Colab kernel's model dependencies or require a restart.

By default it then attempts real Edge inference on an official NVIDIA clip.
Select another variant at the top to test its weights. Supply Hugging Face access
to the chosen model and the PhysicalAI-AV dataset through a Colab `HF_TOKEN`
secret or the masked prompt. Tokens are passed in memory, and the downloadable
report contains logs/results rather than credentials or model caches.

L4/A100 are useful targets for Edge; Nano needs at least 24 GB VRAM. T4 lacks
native BF16, and Super usually exceeds Colab GPU memory. Unsupported GPU/memory
conditions explicitly skip inference, while all adapter tests still run. The
notebook reports failures and skips separately and records the tested Bridge
commit. It starts no CARLA server; an optional short driving test needs your own
CARLA 0.9.16 server reachable from Colab. A dataset prediction alone does not
validate closed-loop driving.
