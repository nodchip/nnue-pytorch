from pathlib import Path
import sys
from types import SimpleNamespace

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
