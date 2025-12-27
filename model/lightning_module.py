import lightning as L
import ranger21
import torch
from torch import Tensor, nn

from .config import LossParams, ModelConfig
from .features import FeatureSet
from .model import NNUEModel
from .quantize import QuantizationConfig


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
        num_batches_per_epoch=int(100_000_000 / 16384),
        gamma=0.992,
        lr=8.75e-4,
        param_index=0,
        num_ls_buckets=9,
        loss_params=LossParams(),
    ):
        super().__init__()
        self.model: NNUEModel = NNUEModel(
            feature_set, config, quantize_config, num_ls_buckets
        )
        self.loss_params = loss_params
        self.max_epoch = max_epoch
        self.num_batches_per_epoch = num_batches_per_epoch
        self.gamma = gamma
        self.lr = lr
        self.param_index = param_index

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def step_(self, batch: tuple[Tensor, ...], batch_idx, loss_type):
        _ = batch_idx

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

        # NNUE raw score (cp-like)
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

        # ---- HalfKP-style win_rate_model ----
        # NNUE output is treated as LOGIT
        q = scorenet / p.out_scaling

        # Search score -> win rate
        pf = torch.sigmoid(score / p.out_scaling)

        # Actual game outcome (0/0.5/1)
        t = outcome

        # Lambda schedule (epoch-based)
        actual_lambda = p.start_lambda + (p.end_lambda - p.start_lambda) * (
            self.current_epoch / self.max_epoch
        )

        # ---- Stockfish-style cross entropy ----
        eps = 1e-12

        teacher_loss = -(
            pf * torch.nn.functional.logsigmoid(q) +
            (1.0 - pf) * torch.nn.functional.logsigmoid(-q)
        )

        outcome_loss = -(
            t * torch.nn.functional.logsigmoid(q) +
            (1.0 - t) * torch.nn.functional.logsigmoid(-q)
        )

        loss = actual_lambda * teacher_loss + (1.0 - actual_lambda) * outcome_loss
        loss = loss.mean()

        self.log(loss_type, loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step_(batch, batch_idx, "train_loss")

    def validation_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, "val_loss")

    def test_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, "test_loss")

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

        optimizer = ranger21.Ranger21(
            train_params,
            lr=1.0,
            betas=(0.9, 0.999),
            eps=1.0e-7,
            using_gc=False,
            using_normgc=False,
            weight_decay=0.0,
            num_batches_per_epoch=self.num_batches_per_epoch,
            num_epochs=self.max_epoch,
            warmdown_active=False,
            use_warmup=False,
            use_adaptive_gradient_clipping=False,
            softplus=False,
            pnm_momentum_factor=0.0,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=self.gamma
        )

        return [optimizer], [scheduler]
