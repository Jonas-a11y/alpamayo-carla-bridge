# Choose and install a model

[Back to README](../README.md) · [Tutorials](TUTORIALS.md)

## Supported driving variants

| Variant | Native model/API | Cameras / images | Ego poses | Navigation in this bridge |
| --- | --- | --- | --- | --- |
| `r1` — Alpamayo 1 Nano | `AlpamayoR1.sample_trajectories_from_data_with_vlm_rollout` | IDs 0, 1, 2, 6; 4 × 4 images | 16 | Not used; omit text |
| `r15` — Alpamayo 1.5 Nano | `Alpamayo1_5.sample_trajectories_from_data_with_vlm_rollout` | IDs 0, 1, 2, 6; 4 × 4 images | 16 | Optional instruction |
| `edge` — Alpamayo 2 Edge | `Alpamayo2Edge.sample_trajectories_from_data` | IDs 0, 1, 2, 3, 5, 6; 22 images | 20 | Required instruction |
| `super` — Alpamayo 2 Super | `Alpamayo2Super.sample_trajectories_from_data` | IDs 0, 1, 2, 3, 5, 6; 6 × 4 images | 16 | Not used in the standard public trajectory path |

All return 64 positions and rotations at 0.1-second intervals. The bridge exposes
candidate zero as XY meters (X forward, Y left) with that candidate's reasoning.
It does not rank Edge/Super's placeholder log probabilities or use future labels.

The upstream names "Alpamayo-R1" and "Alpamayo 1 Nano" refer to the same model
family; the published Hugging Face ID still contains R1. Alpamayo 1.5 support
addresses [issue #1](https://github.com/Jonas-a11y/alpamayo-carla-bridge/issues/1).
The adapter is for driving trajectories; upstream VQA, labeling and training
features are not exposed as CARLA control methods.

## Two runtime environments

Use Python 3.12 for model inference. Edge pins Transformers 5.17.0; the two Nano
models and Super use 4.57.1. Install Edge separately from the Nano/Super runtime.
Do not install both extras into the same environment. The lightweight core can
be installed with Python 3.10 or later and used without either model runtime.

### Edge

~~~bash
python3.12 -m venv .venv-edge
source .venv-edge/bin/activate
pip install -e '.[edge,carla,video]'
hf auth login
~~~

The Edge extra installs its complete native package from the pinned source
revision `02f55a7aeabc1a2b9f7a80445c08e40bddc89f5e`. The default config also pins
the released checkpoint revision. SDPA is selected; FlashAttention is optional.
NVIDIA's reference runtime is Linux/CUDA/BF16 with at least 16 GB VRAM recommended.

### Alpamayo 1, 1.5 and 2 Super

The three packages can share one environment. Installing their code does not
load weights, so this setup lets you select any of the three later:

~~~bash
python3.12 -m venv .venv-nano-super
source .venv-nano-super/bin/activate
pip install -e '.[nano-super,carla,video]'
pip install --no-deps \
  'alpamayo_r1 @ git+https://github.com/NVlabs/alpamayo.git@11a0e01c13a5622377c45ee37d653351453ec43b' \
  'alpamayo1_5 @ git+https://github.com/NVlabs/alpamayo1.5.git@36aeb4c5938cbc2eb2aed33b22434773da4ab639' \
  'alpamayo2_super @ git+https://github.com/NVlabs/alpamayo2.git@59776920bbd981f2c989213bba5ee8594e71ed1d'
hf auth login
~~~

The `nano-super` extra installs the shared native inference dependencies.
`--no-deps` on the source packages avoids their mandatory FlashAttention build
and dataset/notebook dependencies, which this CARLA/SDPA path does not use.
It must follow installation of the runtime extra; it is not a standalone setup.

Nano 1 and 1.5 recommend at least 24 GB VRAM. Super has 34B parameters: the BF16
weights alone are roughly 68 GB before activation/KV-cache overhead. Size the
host from the [Super model card](https://huggingface.co/nvidia/Alpamayo2-Super).
The standard adapter loads the model directly onto the configured device through
`device_map`. It does not implement the upstream advanced two-GPU navigation-CFG
demo or distribute the model automatically across devices.

### Select from CLI or Python

Start each command in the checkout with a compatible CARLA server running:

~~~bash
alpamayo-carla-drive --model r15 --max-steps 10 --no-video
alpamayo-carla-drive --model r1 --max-steps 10 --no-video
alpamayo-carla-drive --model super --max-steps 10 --no-video
~~~

~~~python
from bridge import CarlaBridge, load_config

config = load_config("config.yaml", model="r15")
config["recording"]["enabled"] = False
with CarlaBridge(config) as bridge:
    result = bridge.step(nav_text="Turn right in 30m")
~~~

Changing the model through the selector keeps CARLA, cameras, device, dtype and
recording settings. It clears the previous checkpoint ID/revision and removes
navigation for R1/Super. The inference wrapper then selects that model's default
Hugging Face ID. Re-selecting the same variant preserves its existing checkpoint.
For a local or custom checkpoint, set the variant and checkpoint together:

~~~python
config = load_config("config.yaml", model="r15")
config["alpamayo"]["model_id"] = "/path/to/Alpamayo-1.5-10B"
config["alpamayo"]["local_files_only"] = True
~~~

Nano processors also resolve their Qwen processor assets. Ensure those assets
are cached when using a fully offline environment. `local_files_only` applies to
model loading; it does not turn every upstream helper into an offline loader.

## Capture schedules

| Variant | Image offsets relative to query | Ego history offsets |
| --- | --- | --- |
| R1 / 1.5 / Super | −0.3, −0.2, −0.1, 0 seconds per camera | −1.5 through 0 seconds |
| Edge front wide/tele | −2, −1, −0.5, −0.1, 0 seconds | −1.9 through 0 seconds |
| Edge remaining cameras | −1, −0.1, 0 seconds | Same 20 poses |

The bridge requires synchronous 0.1-second CARLA ticks and captures every tick.
It keeps actual timestamps and never pads missing observations. Edge's helper
handles per-camera pixel budgets, relative-time labels and uint8 normalization;
1.5 and Super's helpers include their native camera/frame labels.

## CPU contract checks

In the Edge environment:

~~~bash
pip install -e '.[edge,test]'
python -m pytest -q -m 'not nano_super'
~~~

In the Nano/Super environment, after installing the three source packages:

~~~bash
pip install -e '.[nano-super,test]'
python -m pytest -q -m 'not edge'
~~~

These commands do not load checkpoints. Runtime-specific tests use the actual
upstream helpers and model-class imports with mocked predictions/processors;
shared tests exercise capture, lifecycle, coordinates and selection behavior.
