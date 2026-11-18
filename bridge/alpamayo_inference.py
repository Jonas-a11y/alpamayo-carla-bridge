"""Native Alpamayo trajectory inference; model dependencies load on demand."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AlpamayoOutput:
    """One candidate: 64 ego-frame XY waypoints at +0.1 through +6.4 seconds."""

    trajectory: list[list[float]]
    reasoning: str


class AlpamayoInference:
    """Load either Alpamayo 1 or Alpamayo 2 Edge and run its action expert.

    Call predict with an ObservationBuffer sample or a dictionary in the
    corresponding upstream dataset format. Imports never download weights.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._cfg = config.get("alpamayo", {})
        self.variant = self._cfg.get("variant", "edge")
        if self.variant not in ("edge", "r1"):
            raise ValueError("alpamayo.variant must be 'edge' or 'r1'")
        default_id = "nvidia/Alpamayo2-Edge" if self.variant == "edge" else "nvidia/Alpamayo-R1-10B"
        self._model_id = self._cfg.get("model_id", default_id)
        self._device = self._cfg.get("device", "cuda")
        self._dtype = self._cfg.get("dtype", "bfloat16")
        if self._dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("alpamayo.dtype must be float32, float16 or bfloat16")
        self._model = None
        self._processor = None
        self._helper = None

    def load(self) -> None:
        import torch

        device = torch.device(self._device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; use a compatible NVIDIA GPU host")
            if self._dtype == "bfloat16":
                with torch.cuda.device(device):
                    if not torch.cuda.is_bf16_supported():
                        raise RuntimeError("The selected CUDA GPU does not support BF16")
        attn = self._cfg.get("attn_implementation", "sdpa")
        resolve = {key: self._cfg[key] for key in ("revision", "local_files_only") if key in self._cfg}
        if self.variant == "edge":
            from alpamayo2_edge import helper
            from alpamayo2_edge.checkpoint import load_model

            model, _ = load_model(
                self._model_id, device=self._device, dtype=self._dtype,
                attn_implementation=attn, **resolve,
            )
        else:
            from alpamayo_r1 import helper
            from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

            model = AlpamayoR1.from_pretrained(
                self._model_id, dtype=getattr(torch, self._dtype),
                attn_implementation=attn, **resolve,
            ).eval().to(self._device)
            self._processor = helper.get_processor(model.tokenizer)
        self._helper = helper
        self._model = model

    def predict(self, sample: dict[str, Any]) -> AlpamayoOutput:
        """Return candidate zero; propagate failures without invented trajectories.

        Selection uses neither future ground truth nor Edge's placeholder log
        probability. Each candidate has its own reasoning trace.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")
        import torch

        data = {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                for key, value in sample.items()}
        if self.variant == "edge":
            inputs = self._helper.prepare_model_inputs(data, self._model.config, self._model.tokenizer)
        else:
            messages = self._helper.create_message(data["image_frames"].flatten(0, 1))
            tokenized = self._processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False,
                continue_final_message=True, return_dict=True, return_tensors="pt",
            )
            inputs = {
                "tokenized_data": tokenized,
                "ego_history_xyz": data["ego_history_xyz"],
                "ego_history_rot": data["ego_history_rot"],
            }
        inputs = self._helper.to_device(inputs, device=self._device)
        device_type = torch.device(self._device).type
        autocast = (torch.autocast(device_type, dtype=getattr(torch, self._dtype))
                    if self._dtype != "float32" else nullcontext())
        settings = {
            "num_traj_samples": self._cfg.get("num_traj_samples", 1),
            "max_generation_length": self._cfg.get("max_generation_length", 256),
            "return_extra": True,
        }
        with torch.inference_mode(), autocast:
            if self.variant == "edge":
                xyz, rotations, _, extra = self._model.sample_trajectories_from_data(inputs, **settings)
            else:
                xyz, rotations, extra = self._model.sample_trajectories_from_data_with_vlm_rollout(
                    data=inputs, **settings,
                )
        expected = (1, 1, settings["num_traj_samples"], 64, 3)
        if tuple(xyz.shape) != expected or tuple(rotations.shape) != (*expected, 3):
            raise ValueError("Model returned unexpected trajectory shapes")
        if not torch.isfinite(xyz).all() or not torch.isfinite(rotations).all():
            raise ValueError("Model returned a non-finite trajectory")
        reasoning = str(extra["cot"][0][0][0])
        return AlpamayoOutput(xyz[0, 0, 0, :, :2].float().cpu().tolist(), reasoning)
