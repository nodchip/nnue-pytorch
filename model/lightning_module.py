import lightning as L
import ranger21
import sys
import torch
from torch import Tensor, nn
from typing import Union, Optional, Callable, Any

from .config import LossParams, ModelConfig
from .features import FeatureSet
from .model import NNUEModel
from .quantize import QuantizationConfig
from torch.optim.optimizer import Optimizer
from lightning.pytorch.core.optimizer import LightningOptimizer


def _get_parameters(layers: list[nn.Module]):
    return [p for layer in layers for p in layer.parameters()]


class NNUE(L.LightningModule):
    """
    feature_set - an instance of FeatureSet defining the input features

    lambda_ = 0.0 - purely based on game results
    0.0 < lambda_ < 1.0 - interpolated score and result
    lambda_ = 1.0 - purely based on search scores

    gamma - the multiplicative factor applied to the learning rate after each epoch

    lr - the initial learning rate
    """

    def __init__(
        self,
        feature_set: FeatureSet,
        config: ModelConfig,
        quantize_config: QuantizationConfig,
        max_epoch=800,
        lr=8.75e-4,
        num_ls_buckets=9,
        loss_params=LossParams(),
        num_batches_warmup=10000,
        newbob_decay=0.5,
        num_epochs_to_adjust_lr=500,
        min_newbob_scale=1e-5,
        momentum=0.0,
    ):
        super().__init__()
        self.model: NNUEModel = NNUEModel(
            feature_set, config, quantize_config, num_ls_buckets
        )
        self.loss_params = loss_params
        self.max_epoch = max_epoch
        self.lr = lr
        self.num_batches_warmup = num_batches_warmup
        self.newbob_scale = 1.0
        self.newbob_decay = newbob_decay
        self.best_loss = 1e10
        self.num_epochs_to_adjust_lr = num_epochs_to_adjust_lr
        self.latest_loss_sum = 0.0
        self.latest_loss_count = 0
        # Warmupを開始するステップ数
        self.warmup_start_global_step = 0
        self.min_newbob_scale = min_newbob_scale
        self.momentum = momentum

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def step_(self, batch: tuple[Tensor, ...], batch_idx, loss_type):
        _ = batch_idx  # unused, but required by pytorch-lightning

        (
            us,
            them,
            white_indices,
            white_values,
            black_indices,
            black_values,
            outcome,
            score,
            layer_stack_indices,
        ) = batch

        scorenet = (
            self.model(
                us,
                them,
                white_indices,
                white_values,
                black_indices,
                black_values,
                layer_stack_indices,
            )
            * self.model.quantization.nnue2score
        )

        p = self.loss_params
        # convert the network and search scores to an estimate match result
        # based on the win_rate_model, with scalings and offsets optimized
        q = (scorenet - p.in_offset) / p.in_scaling
        qm = (-scorenet - p.in_offset) / p.in_scaling
        qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())

        s = (score - p.out_offset) / p.out_scaling
        sm = (-score - p.out_offset) / p.out_scaling
        pf = 0.5 * (1.0 + s.sigmoid() - sm.sigmoid())

        # blend that eval based score with the actual game outcome
        t = outcome
        actual_lambda = p.start_lambda + (p.end_lambda - p.start_lambda) * (
            self.current_epoch / self.max_epoch
        )
        pt = pf * actual_lambda + t * (1.0 - actual_lambda)

        # use a MSE-like loss function
        loss = torch.pow(torch.abs(pt - qf), p.pow_exp)
        if p.qp_asymmetry != 0.0:
            loss = loss * ((qf > pt) * p.qp_asymmetry + 1)
        loss = loss.mean()

        self.log(loss_type, loss, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step_(batch, batch_idx, "train_loss")

    def validation_step(self, batch, batch_idx):
        outputs = self.step_(batch, batch_idx, "val_loss")
        self.latest_loss_sum += outputs.item()
        self.latest_loss_count += 1

    def on_validation_epoch_end(self):
        if self.newbob_decay == 1.0:
            return

        if self.current_epoch == 0:
            return

        if self.current_epoch % self.num_epochs_to_adjust_lr != 0:
            return

        latest_loss = self.latest_loss_sum / self.latest_loss_count
        self.latest_loss_sum = 0.0
        self.latest_loss_count = 0
        if latest_loss < self.best_loss:
            self.print(
                f"{self.current_epoch=}, {latest_loss=} < {self.best_loss=}, accepted, {self.newbob_scale=}"
            )
            sys.stdout.flush()
            self.best_loss = latest_loss
        else:
            self.newbob_scale *= self.newbob_decay
            self.print(
                f"{self.current_epoch=}, {latest_loss=} >= {self.best_loss=}, rejected, {self.newbob_scale=}"
            )
            sys.stdout.flush()

        if self.newbob_scale < self.min_newbob_scale:
            self.trainer.should_stop = True
            self.print(f"{self.current_epoch=}, early stopping")

    def test_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, "test_loss")

    # learning rate warm-up
    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Union[Optimizer, LightningOptimizer],
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ):
        # manually warm up lr without a scheduler
        if (
            self.trainer.global_step - self.warmup_start_global_step
            < self.num_batches_warmup
        ):
            warmup_scale = min(
                1.0,
                float(self.trainer.global_step - self.warmup_start_global_step + 1)
                / self.num_batches_warmup,
            )
        else:
            warmup_scale = 1.0
        for pg in optimizer.param_groups:
            pg["lr"] = self.lr * warmup_scale * self.newbob_scale
            self.log("lr", pg["lr"])

        # update params
        optimizer.step(closure=optimizer_closure)

    def configure_optimizers(self):
        LR = self.lr
        train_params = [
            {"params": _get_parameters([self.model.input]), "lr": LR, "gc_dim": 0},
            {"params": [self.model.layer_stacks.l1.factorized_linear.weight], "lr": LR},
            {"params": [self.model.layer_stacks.l1.factorized_linear.bias], "lr": LR},
            {"params": [self.model.layer_stacks.l1.linear.weight], "lr": LR},
            {"params": [self.model.layer_stacks.l1.linear.bias], "lr": LR},
            {"params": [self.model.layer_stacks.l2.linear.weight], "lr": LR},
            {"params": [self.model.layer_stacks.l2.linear.bias], "lr": LR},
            {"params": [self.model.layer_stacks.output.linear.weight], "lr": LR},
            {"params": [self.model.layer_stacks.output.linear.bias], "lr": LR},
        ]

        return torch.optim.SGD(train_params, lr=self.lr, momentum=self.momentum)
