import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from bridge import AlpamayoInference
from test_observation import fill_buffer


class EdgeModel:
    config = SimpleNamespace()
    tokenizer = object()

    def sample_trajectories_from_data(self, data, **settings):
        assert data["ego_history_xyz"].shape == (1, 20, 3)
        assert data["tokenized_data"]["input_ids"].dtype == torch.int64
        count = settings["num_traj_samples"]
        xyz = torch.zeros(1, 1, count, 64, 3)
        xyz[0, 0, 0, :, 0] = torch.arange(1, 65) / 10
        rot = torch.eye(3).expand(1, 1, count, 64, 3, 3).clone()
        # Log probabilities are placeholders and intentionally favor candidate 1.
        return xyz, rot, torch.arange(count), {"cot": np.array([[['first', 'second'][:count]]])}


def loaded_edge(monkeypatch, model=None):
    from alpamayo2_edge import helper, checkpoint
    from alpamayo2_edge.sample import validate_sample

    model = model or EdgeModel()
    calls = []

    def load(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return model, {}

    def prepare(data, config, tokenizer):
        validate_sample(data)
        return {"tokenized_data": {"input_ids": torch.tensor([[1, 2]])},
                "ego_history_xyz": data["ego_history_xyz"]}

    monkeypatch.setattr(checkpoint, "load_model", load)
    monkeypatch.setattr(helper, "prepare_model_inputs", prepare)
    inference = AlpamayoInference({"alpamayo": {"variant": "edge", "device": "cpu",
                    "dtype": "float32", "revision": "immutable", "num_traj_samples": 2}})
    inference.load()
    return inference, calls


@pytest.mark.edge
def test_edge_loader_native_sampling_and_candidate_selection(monkeypatch):
    inference, calls = loaded_edge(monkeypatch)
    inference.load()
    result = inference.predict(fill_buffer().sample("Follow lane"))
    assert calls == [("nvidia/Alpamayo2-Edge", {"device": "cpu", "dtype": "float32",
                      "attn_implementation": "sdpa", "revision": "immutable"})]
    np.testing.assert_allclose(result.trajectory[0], [0.1, 0])
    assert len(result.trajectory) == 64
    assert result.reasoning == "first"


@pytest.mark.edge
@pytest.mark.parametrize("failure", ["exception", "nan", "shape"])
def test_inference_failures_do_not_invent_a_driving_trajectory(monkeypatch, failure):
    class BrokenModel(EdgeModel):
        def sample_trajectories_from_data(self, data, **settings):
            if failure == "exception":
                raise RuntimeError("model failed")
            xyz, rot, probs, extra = super().sample_trajectories_from_data(data, **settings)
            if failure == "nan":
                rot[0, 0, 0, 0, 0, 0] = float("nan")
            else:
                xyz = xyz[..., :10, :]
            return xyz, rot, probs, extra
    inference, _ = loaded_edge(monkeypatch, BrokenModel())
    with pytest.raises((RuntimeError, ValueError)):
        inference.predict(fill_buffer().sample("Follow lane"))


def test_r1_uses_native_loader_prompt_and_rollout(monkeypatch):
    package = ModuleType("alpamayo_r1")
    module = ModuleType("alpamayo_r1.models.alpamayo_r1")
    seen = {}

    class R1Model:
        tokenizer = object()

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            seen["load"] = (model_id, kwargs)
            return cls()

        def eval(self):
            return self

        def to(self, device):
            seen["device"] = device
            return self

        def sample_trajectories_from_data_with_vlm_rollout(self, data, **settings):
            assert data["ego_history_xyz"].shape == (1, 1, 16, 3)
            assert data["ego_history_rot"].shape == (1, 1, 16, 3, 3)
            return torch.zeros(1, 1, 1, 64, 3), torch.eye(3).expand(1, 1, 1, 64, 3, 3), {"cot": np.array([[['r1 reasoning']]])}

    def prompt(frames):
        assert frames.shape == (16, 3, 4, 6)
        return "native prompt"

    def tokenize(messages, **kwargs):
        assert messages == "native prompt"
        assert kwargs["continue_final_message"] is True
        assert kwargs["add_generation_prompt"] is False
        return {"input_ids": torch.tensor([[1]])}

    package.helper = SimpleNamespace(create_message=prompt,
        get_processor=lambda tokenizer: SimpleNamespace(apply_chat_template=tokenize),
        to_device=lambda data, device: data)
    module.AlpamayoR1 = R1Model
    monkeypatch.setitem(sys.modules, "alpamayo_r1", package)
    monkeypatch.setitem(sys.modules, "alpamayo_r1.models", ModuleType("alpamayo_r1.models"))
    monkeypatch.setitem(sys.modules, "alpamayo_r1.models.alpamayo_r1", module)
    inference = AlpamayoInference({"alpamayo": {"variant": "r1", "device": "cpu", "dtype": "float32"}})
    inference.load()
    result = inference.predict(fill_buffer("r1").sample())
    assert seen["load"] == ("nvidia/Alpamayo-R1-10B", {"dtype": torch.float32, "attn_implementation": "sdpa"})
    assert seen["device"] == "cpu"
    assert result.reasoning == "r1 reasoning"
