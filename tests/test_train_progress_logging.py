from pathlib import Path
import io
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace

from lightning.pytorch.callbacks import TQDMProgressBar
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import (
    LineProgressCallback,
    build_progress_callbacks,
    format_progress_line,
    resolve_progress_mode,
)


def test_resolve_progress_mode_uses_log_for_non_tty():
    assert resolve_progress_mode("auto", is_tty=False) == "log"
    assert resolve_progress_mode("auto", is_tty=True) == "tqdm"


def test_build_progress_callbacks_uses_line_logger_for_log_mode():
    callbacks, enable_progress_bar = build_progress_callbacks(
        progress_mode="log",
        progress_log_interval=1,
        batch_size=16384,
        is_tty=False,
    )

    assert enable_progress_bar is False
    assert any(isinstance(callback, LineProgressCallback) for callback in callbacks)
    assert all(not isinstance(callback, TQDMProgressBar) for callback in callbacks)


def test_format_progress_line_emits_fixed_key_value_order():
    message = format_progress_line(
        phase="train",
        epoch=12,
        total_epochs=1200,
        step=18432,
        positions=301989888,
        lr=4.375e-4,
        loss=0.512384,
        val_loss=0.498221,
        elapsed_seconds=3 * 3600 + 14 * 60 + 52,
        eta_seconds=8 * 3600 + 21 * 60 + 10,
        speed=15432.7,
    )

    assert (
        message
        == "progress phase=train epoch=12/1200 step=18432 positions=301989888 "
        "lr=0.00043750 loss=0.512384 val_loss=0.498221 elapsed=03:14:52 "
        "eta=08:21:10 speed=15432.7pos/s"
    )


def test_line_progress_callback_logs_on_train_step_interval():
    callback = LineProgressCallback(log_every_n_steps=25, batch_size=16384)
    trainer = SimpleNamespace(
        optimizers=[torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.0))], lr=4.375e-4)],
        current_epoch=0,
        max_epochs=2,
        global_step=25,
        estimated_stepping_batches=100,
        callback_metrics={"val_loss": torch.tensor(0.25)},
    )

    callback.start_time = 0.0
    train_buf = io.StringIO()
    with redirect_stdout(train_buf):
        callback.on_train_batch_end(trainer, None, torch.tensor(0.5), None, 0)
    assert "progress phase=train epoch=1/2 step=25 positions=409600" in train_buf.getvalue()


def test_line_progress_callback_skips_sanity_check_validation():
    callback = LineProgressCallback(log_every_n_steps=25, batch_size=16384)
    trainer = SimpleNamespace(
        optimizers=[torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.0))], lr=4.375e-4)],
        current_epoch=0,
        max_epochs=2,
        global_step=0,
        estimated_stepping_batches=100,
        callback_metrics={"val_loss": torch.tensor(0.25)},
        sanity_checking=True,
    )

    callback.start_time = 0.0
    buf = io.StringIO()
    with redirect_stdout(buf):
        callback.on_validation_epoch_end(trainer, None)
    assert buf.getvalue() == ""
