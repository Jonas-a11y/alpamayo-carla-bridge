import pytest

from bridge import load_config


@pytest.mark.parametrize("model", ["r1", "r15", "super"])
def test_switching_models_resets_checkpoint_and_keeps_shared_runtime_settings(model):
    config = load_config(model=model)
    assert config["alpamayo"]["variant"] == model
    assert "model_id" not in config["alpamayo"]
    assert "revision" not in config["alpamayo"]
    assert config["alpamayo"]["device"] == "cuda"
    assert config["carla"]["fixed_delta"] == 0.1
    assert ("nav_text" in config["alpamayo"]) == (model == "r15")


def test_selecting_existing_model_preserves_pinned_checkpoint():
    assert load_config(model="edge") == load_config()
