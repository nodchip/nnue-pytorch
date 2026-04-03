from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.optuna_tune_train import build_train_command, extract_val_loss
from scripts.optuna_tune_train import suggest_trial_params


def test_extract_val_loss_reads_progress_log_line():
    line = (
        "progress phase=val epoch=2/10 step=400 positions=3276800 lr=0.00080000 "
        "loss=na val_loss=0.123456 elapsed=00:10:00 eta=00:20:00 speed=1234.5pos/s"
    )

    assert extract_val_loss(line) == pytest.approx(0.123456)


def test_build_train_command_overrides_search_params():
    command = build_train_command(
        python_executable="venv\\Scripts\\python.exe",
        train_script="train.py",
        base_args=["dummy.binpack", "--features", "HalfKA_hm"],
        trial_params={
            "lr": 8.2e-4,
            "gamma": 0.993,
            "beta1": 0.88,
            "beta2": 0.997,
        },
    )

    assert command[:2] == ["venv\\Scripts\\python.exe", "train.py"]
    assert "--progress-mode" in command
    assert command[command.index("--progress-mode") + 1] == "log"
    assert command[command.index("--lr") + 1] == "0.00082"
    assert command[command.index("--gamma") + 1] == "0.993"
    assert command[command.index("--beta1") + 1] == "0.88"
    assert command[command.index("--beta2") + 1] == "0.997"


def test_suggest_trial_params_uses_cli_ranges():
    class DummyTrial:
        def __init__(self):
            self.calls = []

        def suggest_float(self, name, low, high, log=False):
            self.calls.append((name, low, high, log))
            return low

    class DummyArgs:
        lr_min = 1.0e-4
        lr_max = 9.0e-4
        gamma_min = 0.991
        gamma_max = 0.996
        beta1_min = 0.86
        beta1_max = 0.93
        beta2_min = 0.996
        beta2_max = 0.999

    trial = DummyTrial()

    params = suggest_trial_params(trial, DummyArgs())

    assert params == {
        "lr": pytest.approx(1.0e-4),
        "gamma": pytest.approx(0.991),
        "beta1": pytest.approx(0.86),
        "beta2": pytest.approx(0.996),
    }
    assert trial.calls == [
        ("lr", 1.0e-4, 9.0e-4, True),
        ("gamma", 0.991, 0.996, False),
        ("beta1", 0.86, 0.93, False),
        ("beta2", 0.996, 0.999, False),
    ]
