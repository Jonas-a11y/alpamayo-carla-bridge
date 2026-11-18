# Troubleshooting

[README](../README.md) · [Tutorials](TUTORIALS.md) · [Model setup](MODELS.md)

Start with the first traceback from a failed CLI run. The bridge propagates model
and capture errors and commands braking on a failed step. Correct the cause
before starting a new session.

| Symptom | Check and action |
| --- | --- |
| `ModuleNotFoundError: bridge` / CLI missing | Activate the environment where you installed the checkout. Run `python -m pip install -e .` there, then `python -c 'import bridge; print(bridge.__file__)'`. |
| `config.yaml` not found | The default path is relative to the current directory. Pass `--config /path/to/config.yaml` or an explicit path to `load_config`. |
| Missing `alpamayo2_edge`, `alpamayo_r1`, `alpamayo1_5` or `alpamayo2_super` | Install the selected runtime and its pinned upstream source. The core alone intentionally contains no model code. Nano/Super setup has a separate source-install step. |
| Transformers conflict or FlashAttention build error | Use separate Edge and Nano/Super environments. Follow the SDPA installation order in [model setup](MODELS.md); install the Nano/Super code with `--no-deps` after its runtime extra. |
| No matching `carla==0.9.16` distribution on macOS | Use Linux x86_64 or Windows with Python 3.12 for the published client wheel. The laptop core can still import; a remote server alone does not supply the Python client wheel. |
| Connection timeout | Start CARLA, confirm `host`/`port` and reachable server ports. Match client/server versions. Increase `carla.timeout` only when the server is reachable but slow. |
| Configured map unavailable | Install the corresponding CARLA maps or select an available map. The bridge logs a warning and uses the current world when the requested map is unavailable. |
| CUDA unavailable / BF16 unsupported | Check `torch.cuda.is_available()`, driver/runtime compatibility and GPU selection. Use the model's supported NVIDIA host; setting CPU only enables contract tests, not a practical substitute for the reference driving runtime. |
| GPU out of memory | Check the selected model and free device memory. Edge recommends ≥16 GB VRAM, Nano ≥24 GB; Super's BF16 weights alone are roughly 68 GB. This adapter uses one configured device. |
| Hugging Face 401/403 or checkpoint unavailable | Authenticate with `hf auth login`, obtain any required model access, and verify the checkpoint ID/revision for the selected variant. |
| Offline loading still requests assets | Cache the upstream Qwen processor assets as well as model files. `local_files_only` is passed to model loading; it does not constrain all upstream processor helpers. |
| Edge navigation error | Set a nonempty `alpamayo.nav_text` string before startup. Per-step text overrides it. R1/Super instead require navigation to be omitted; 1.5 accepts optional text. |
| Missing camera / timeout / skipped frame / timestamp mismatch | Use the complete model rig and synchronous 0.1 s capture. Stop other clients that tick the world, retain rendering for RGB sensors, and inspect camera previews. Histories cannot be padded with duplicate frames. |
| Unexpected/non-finite trajectory | The native action expert returned an invalid output. Confirm upstream source/checkpoint compatibility and dtype/device settings; the bridge stops instead of substituting a path. |
| Collision detected | The session stops. Inspect the recording, spawn location and calibration before retrying. Restart resets collision state and captures fresh history. |
| `PIL`/`matplotlib` missing or ffmpeg not found | Install `.[video]` and ffmpeg. `--no-video` skips composition; recording still needs Pillow. Set `recording.enabled: false` for a drive without recording. |

For an unresolved problem, include the selected variant, upstream source and
checkpoint revisions, Python/PyTorch/Transformers/CARLA versions, GPU and the
first traceback in a [GitHub issue](https://github.com/Jonas-a11y/alpamayo-carla-bridge/issues).
Describe whether it occurred during load, warmup, prediction or video composition.
CPU tests and a successful import cannot establish model quality or simulator
compatibility on a different host.
