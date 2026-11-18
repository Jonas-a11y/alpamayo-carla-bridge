import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bridge import AlpamayoInference
from test_observation import fill_buffer


@pytest.mark.nano_super
@pytest.mark.parametrize("variant,package,class_name", [
    ("r1", "alpamayo_r1", "AlpamayoR1"),
    ("r15", "alpamayo1_5", "Alpamayo1_5"),
    ("super", "alpamayo2_super", "Alpamayo2Super"),
])
def test_real_native_helpers_and_model_imports(monkeypatch, variant, package, class_name):
    helper = importlib.import_module(package + ".helper")
    module = importlib.import_module(package + ".models." + package)
    model_class = getattr(module, class_name)
    seen = {}

    class Processor:
        def apply_chat_template(self, messages, tokenize, **kwargs):
            seen["messages"] = messages
            seen["template"] = kwargs
            return {"input_ids": torch.tensor([[1, 2]])} if tokenize else "native prompt"

        def __call__(self, **kwargs):
            images = kwargs["images"]
            assert images.shape == (24, 3, 4, 6)
            assert images.dtype == torch.float32
            assert 0 <= images.min() <= images.max() <= 1
            assert kwargs["do_rescale"] is False
            return {"input_ids": torch.tensor([[1, 2]])}

    class Model:
        config = SimpleNamespace(tokens_per_history_traj=48, tokens_per_future_traj=192,
                                 include_camera_ids=True, frame_label="frame_num")
        tokenizer = object()

        def eval(self):
            return self

        def to(self, device):
            assert device == "cpu"
            return self

        def result(self, data, **settings):
            assert data["ego_history_xyz"].shape == (1, 1, 16, 3)
            assert data["ego_history_rot"].shape == (1, 1, 16, 3, 3)
            assert data["tokenized_data"]["input_ids"].dtype == torch.int64
            assert settings["return_extra"] is True
            xyz = torch.zeros(1, 1, 1, 64, 3)
            rot = torch.eye(3).expand(1, 1, 1, 64, 3, 3)
            extra = {"cot": np.array([[['native reasoning']]])}
            return xyz, rot, extra

        def sample_trajectories_from_data_with_vlm_rollout(self, data, **settings):
            assert variant in ("r1", "r15")
            return self.result(data, **settings)

        def sample_trajectories_from_data(self, data, **settings):
            assert variant == "super"
            xyz, rot, extra = self.result(data, **settings)
            return xyz, rot, torch.zeros(1), extra

    def load(model_id, **kwargs):
        seen["model_id"] = model_id
        seen["load"] = kwargs
        return Model()

    monkeypatch.setattr(model_class, "from_pretrained", load)
    monkeypatch.setattr(helper, "get_processor", lambda *args: Processor())
    inference = AlpamayoInference({"alpamayo": {"variant": variant, "device": "cpu", "dtype": "float32"}})
    inference.load()
    nav = "Turn right in 30m" if variant == "r15" else ""
    output = inference.predict(fill_buffer(variant).sample(nav))
    assert len(output.trajectory) == 64
    assert output.reasoning == "native reasoning"
    assert seen["model_id"] == {"r1": "nvidia/Alpamayo-R1-10B", "r15": "nvidia/Alpamayo-1.5-10B",
                                "super": "nvidia/Alpamayo2-Super"}[variant]
    if variant == "super":
        assert seen["load"]["device_map"] == "cpu"
    content = seen["messages"][1]["content"]
    assert sum(item["type"] == "image" for item in content) == (24 if variant == "super" else 16)
    text = " ".join(item.get("text", "") for item in content)
    if variant == "r15":
        assert "<|route_start|>Turn right in 30m<|route_end|>" in text
        assert "Front left camera" in text and "Front telephoto camera" in text
    elif variant == "super":
        from alpamayo2_super.input_profiles import DRIVING_SIX_CAMERA_FOUR_FRAME
        assert tuple(fill_buffer(variant).sample()["camera_indices"]) == DRIVING_SIX_CAMERA_FOUR_FRAME.camera_ids
        assert "Rear left camera" in text and "Rear right camera" in text
