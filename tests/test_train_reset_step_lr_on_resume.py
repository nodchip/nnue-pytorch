from pathlib import Path
from types import SimpleNamespace
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import ResetStepLROnResume


def test_reset_step_lr_on_resume_resets_live_optimizer_and_scheduler_state():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    optimizer.step()
    scheduler.step()
    optimizer.step()
    scheduler.step()

    assert optimizer.param_groups[0]["lr"] == 0.025
    assert scheduler.last_epoch == 2
    assert scheduler._step_count == 3

    trainer = SimpleNamespace(
        optimizers=[optimizer],
        lr_scheduler_configs=[SimpleNamespace(scheduler=scheduler)],
    )

    callback = ResetStepLROnResume(enabled=True, initial_lr=0.1, gamma=0.5)

    callback.on_fit_start(trainer, None)

    assert optimizer.param_groups[0]["lr"] == 0.1
    assert optimizer.param_groups[0]["initial_lr"] == 0.1
    assert scheduler.base_lrs == [0.1]
    assert scheduler.last_epoch == 0
    assert scheduler._step_count == 1
    assert scheduler.get_last_lr() == [0.1]
