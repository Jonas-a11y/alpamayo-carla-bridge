# Alpamayo–CARLA Bridge

Connect NVIDIA's Alpamayo driving models to CARLA through an importable Python
package. A model decision returns 64 ego-frame waypoints covering +0.1 through
+6.4 seconds, its reasoning, and the resulting throttle/steering/brake command.
The CLI and Python API use the same synchronous driving implementation.

| Model | Select with | Checkpoint |
| --- | --- | --- |
| Alpamayo 1 Nano (formerly R1) | `r1` | `nvidia/Alpamayo-R1-10B` |
| Alpamayo 1.5 Nano | `r15` | `nvidia/Alpamayo-1.5-10B` |
| Alpamayo 2 Edge — default | `edge` | `nvidia/Alpamayo2-Edge` |
| Alpamayo 2 Super | `super` | `nvidia/Alpamayo2-Super` |

This project runs their native **trajectory/action-expert APIs**. It does not
parse coordinates from ordinary text generation or replace failed predictions
with an invented driving path. Additional upstream tasks such as VQA, labeling,
training and advanced multi-GPU navigation CFG are outside this driving API.

## Start here

- [Tutorials: laptop setup, first CARLA run and Python usage](docs/TUTORIALS.md)
- [CARLA-only Colab notebook: test server startup, RGB rendering and a three-second drive first](notebooks/carla_colab.ipynb)
- [Colab notebook: check all adapters, try GPU inference and optionally render video](notebooks/alpamayo_colab.ipynb)
- [Model selection, installations and input profiles](docs/MODELS.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Camera placement and rear-axle calibration](docs/CAMERA_CONFIG.md)
- [Architecture and lifecycle](docs/ARCHITECTURE.md)

### Install the importable core

From a checkout, on macOS, Linux or Windows:

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
alpamayo-carla-drive --help
~~~

On Windows, activate with `.venv/Scripts/Activate.ps1` in PowerShell. The
installed distribution is `alpamayo-carla-bridge` and the Python import is
`bridge`. The core needs only NumPy and PyYAML; importing it does not load
PyTorch, Transformers, CARLA, model weights or a network connection.

### First Edge drive on the target host

Use **Python 3.12**, an NVIDIA CUDA GPU with BF16 support, matching CARLA
0.9.16 server/client versions, and Hugging Face access to the checkpoint:

~~~bash
pip install -e '.[edge,carla,video]'
hf auth login
# Start CARLA 0.9.16 in another terminal before this command.
alpamayo-carla-drive --config config.yaml --model edge --max-steps 10 --no-video
~~~

Edge's BF16 reference setup recommends at least 16 GB VRAM. Nano needs at
least 24 GB; Super is a 34B model and requires substantially more memory. See
[model setup](docs/MODELS.md) before changing variants. The CARLA extra has
Python 3.12 wheels for Linux x86_64 and Windows, not macOS. Optional video
composition requires ffmpeg on PATH.

### Use it from Python

~~~python
from bridge import CarlaBridge, load_config

config = load_config("config.yaml", model="edge")
config["recording"]["enabled"] = False

with CarlaBridge(config) as bridge:
    for _ in range(10):
        result = bridge.step(nav_text="Follow the current lane.")
        print(result.state.timestamp_us, result.output.trajectory[0])
        print(result.output.reasoning)
~~~

For Alpamayo 1.5 use `model="r15"` in its installed Nano/Super environment.
For R1 or Super, call `bridge.step()` without navigation text. Selecting a
*different* model through `load_config(..., model=...)` or `--model` resets the
previous checkpoint ID/revision and removes navigation for variants that do
not use it. Selecting the existing variant preserves its pinned/custom checkpoint.

The context manager loads the model, spawns the rig, captures real history while
braking, and restores actors/world settings on exit. Restarting the same bridge
captures a new history and reuses its loaded model. Inference, capture and
collision failures raise; a failed step commands braking. CLI failures exit
nonzero.

## Verification and current limits

~~~bash
pip install -e '.[edge,test,video]'
ffmpeg -version
python -m pytest -q -m 'not nano_super'
~~~

CI checks both the Edge and Nano/Super runtimes. Tests exercise actual upstream
sample/prompt helpers and model-class imports, mocked action-expert predictions,
image schedules, pose conversion, frame synchronization, model selection,
restart behavior, cleanup failures and Colab worker/video regressions. Install
ffmpeg (including ffprobe) for the video checks. See the model guide to run the
Nano/Super checks locally. Tests and core installation do not require model weights.

Edge checkpoint inference passed on one official NVIDIA clip in Colab with an
L4 and BF16: 64 finite XY waypoints, 5.03 s inference and 6.18 GB peak allocated
GPU memory for one candidate. R1/1.5/Super checkpoint runs and closed-loop CARLA
driving remain unverified. A follow-up Colab run confirmed headless EGL/Vulkan
on the L4, but CARLA still did not reach RPC readiness after starting the
configured map directly. Use the CARLA-only notebook to separate engine/RPC
startup from GPU camera rendering before downloading model weights. Its real
Colab server/camera tests remain unverified.
CPU checks and a single dataset clip do not establish driving quality or
real-time performance. CARLA imagery differs from the training data.
The camera rig and approximate rear-axle origin require target-vehicle calibration.

## Sources

- [Alpamayo 1 Nano](https://github.com/NVlabs/alpamayo)
- [Alpamayo 1.5 Nano](https://github.com/NVlabs/alpamayo1.5)
- [Alpamayo 2 Edge](https://github.com/NVlabs/alpamayo2-edge)
- [Alpamayo 2 Super](https://github.com/NVlabs/alpamayo2)
- [CARLA documentation](https://carla.readthedocs.io/)

MIT project; model weights and dependencies have their own licenses.
