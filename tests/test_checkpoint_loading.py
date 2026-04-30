from collections import OrderedDict

import model.lightning_module as lightning_module
from model.utils import checkpoint
from model.utils.checkpoint import (
    load_nnue_from_checkpoint,
    strip_compiled_model_state_dict_keys,
)


def test_strip_compiled_model_state_dict_keys_removes_duplicate_compiled_weights():
    state_dict = OrderedDict(
        [
            ("model.input.weight", "eager-weight"),
            ("_compiled_model._orig_mod.input.weight", "compiled-weight"),
            ("model.input.bias", "eager-bias"),
            ("_compiled_model._orig_mod.input.bias", "compiled-bias"),
        ]
    )

    removed = strip_compiled_model_state_dict_keys(state_dict)

    assert removed == 2
    assert state_dict == {
        "model.input.weight": "eager-weight",
        "model.input.bias": "eager-bias",
    }


def test_strip_compiled_model_state_dict_keys_restores_model_weights_when_needed():
    state_dict = OrderedDict(
        [
            ("_compiled_model._orig_mod.input.weight", "compiled-weight"),
            ("_compiled_model._orig_mod.input.bias", "compiled-bias"),
        ]
    )

    removed = strip_compiled_model_state_dict_keys(state_dict)

    assert removed == 2
    assert state_dict == {
        "model.input.weight": "compiled-weight",
        "model.input.bias": "compiled-bias",
    }


def test_load_nnue_from_checkpoint_strips_compiled_cache_keys(monkeypatch):
    checkpoint_data = {
        "state_dict": OrderedDict(
            [
                ("model.input.weight", "eager-weight"),
                ("_compiled_model._orig_mod.input.weight", "compiled-weight"),
            ]
        )
    }
    load_calls = []

    def fake_torch_load(path, map_location, weights_only):
        load_calls.append((path, map_location, weights_only))
        return checkpoint_data

    class FakeNNUE:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.loaded_state_dict = None

        def load_state_dict(self, state_dict):
            self.loaded_state_dict = state_dict

    monkeypatch.setattr(checkpoint.torch, "load", fake_torch_load)
    monkeypatch.setattr(lightning_module, "NNUE", FakeNNUE)

    loaded_model = load_nnue_from_checkpoint(
        "network.ckpt",
        feature_set="features",
        config="config",
        quantize_config="quantize",
        map_location="cpu",
    )

    assert load_calls == [("network.ckpt", "cpu", False)]
    assert loaded_model.kwargs == {
        "feature_set": "features",
        "config": "config",
        "quantize_config": "quantize",
    }
    assert loaded_model.loaded_state_dict == {
        "model.input.weight": "eager-weight",
    }
