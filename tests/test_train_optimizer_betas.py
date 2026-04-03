from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import model as M
import model.lightning_module as lightning_module
import train


def test_build_arg_parser_accepts_beta_overrides():
    parser = train.build_arg_parser()

    args = parser.parse_args(
        [
            "dummy.binpack",
            "--features",
            "HalfKA_hm",
            "--beta1",
            "0.85",
            "--beta2",
            "0.995",
        ]
    )

    assert args.beta1 == pytest.approx(0.85)
    assert args.beta2 == pytest.approx(0.995)


def test_configure_optimizers_uses_configured_betas(monkeypatch):
    captured = {}

    class DummyOptimizer:
        def __init__(self, param_groups, **kwargs):
            captured["param_groups"] = param_groups
            captured["kwargs"] = kwargs
            self.param_groups = [{"lr": kwargs["lr"]}]

    class DummyScheduler:
        def __init__(self, optimizer, step_size, gamma):
            captured["scheduler"] = {
                "optimizer": optimizer,
                "step_size": step_size,
                "gamma": gamma,
            }

    monkeypatch.setattr(lightning_module.ranger21, "Ranger21", DummyOptimizer)
    monkeypatch.setattr(
        lightning_module.torch.optim.lr_scheduler, "StepLR", DummyScheduler
    )

    nnue = M.NNUE(
        feature_set=M.get_feature_set_from_name("HalfKA_hm"),
        config=M.ModelConfig(),
        quantize_config=M.QuantizationConfig(),
        beta1=0.85,
        beta2=0.995,
    )

    optimizers, schedulers = nnue.configure_optimizers()

    assert len(optimizers) == 1
    assert len(schedulers) == 1
    assert captured["kwargs"]["betas"] == pytest.approx((0.85, 0.995))
    assert captured["scheduler"]["gamma"] == pytest.approx(nnue.gamma)
