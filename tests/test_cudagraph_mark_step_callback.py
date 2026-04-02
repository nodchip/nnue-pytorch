from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.callbacks import CUDAGraphMarkStepCallback


def test_cudagraph_mark_step_callback_marks_train_and_validation_steps(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(
        torch.compiler,
        "cudagraph_mark_step_begin",
        lambda: calls.append("mark"),
    )

    callback = CUDAGraphMarkStepCallback(enabled=True)

    callback.on_train_batch_start(SimpleNamespace(), SimpleNamespace(), None, 0)
    callback.on_validation_batch_start(SimpleNamespace(), SimpleNamespace(), None, 0)

    assert calls == ["mark", "mark"]


def test_cudagraph_mark_step_callback_is_noop_when_disabled(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(
        torch.compiler,
        "cudagraph_mark_step_begin",
        lambda: calls.append("mark"),
    )

    callback = CUDAGraphMarkStepCallback(enabled=False)

    callback.on_train_batch_start(SimpleNamespace(), SimpleNamespace(), None, 0)
    callback.on_validation_batch_start(SimpleNamespace(), SimpleNamespace(), None, 0)

    assert calls == []
