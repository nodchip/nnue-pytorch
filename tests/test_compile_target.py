from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train
from model.lightning_module import NNUE


def test_compile_nnue_model_compiles_wrapped_model_only(monkeypatch):
    calls = []
    compiled_model = object()
    eager_model = object()
    nnue = SimpleNamespace(model=eager_model)

    def fake_compile(target, backend):
        calls.append((target, backend))
        return compiled_model

    monkeypatch.setattr(train.torch, "compile", fake_compile)

    returned = train.compile_nnue_model(nnue, backend="inductor")

    assert returned is nnue
    assert nnue.model is eager_model
    assert nnue._compiled_model is compiled_model
    assert calls == [(eager_model, "inductor")]


def test_forward_model_uses_compiled_model_only_when_grad_is_enabled():
    eager_model = object()
    compiled_model = object()
    fake_nnue = SimpleNamespace(model=eager_model, _compiled_model=compiled_model)

    assert NNUE._forward_model(fake_nnue) is compiled_model

    with train.torch.no_grad():
        assert NNUE._forward_model(fake_nnue) is eager_model


def test_compile_nnue_model_does_not_register_compiled_model_as_child_module(monkeypatch):
    compiled_model = torch.nn.Linear(1, 1)

    class DummyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(1, 1)
            self._compiled_model = None

    nnue = DummyModule()

    monkeypatch.setattr(train.torch, "compile", lambda target, backend: compiled_model)

    train.compile_nnue_model(nnue, backend="inductor")

    assert nnue._compiled_model is compiled_model
    assert "_compiled_model" not in nnue._modules


def test_on_load_checkpoint_discards_compiled_model_state():
    checkpoint = {
        "state_dict": {
            "model.weight": torch.tensor([1.0]),
            "_compiled_model._orig_mod.weight": torch.tensor([2.0]),
            "_compiled_model._orig_mod.bias": torch.tensor([3.0]),
        }
    }

    NNUE.on_load_checkpoint(SimpleNamespace(), checkpoint)

    assert checkpoint["state_dict"] == {"model.weight": torch.tensor([1.0])}
