import lightning as L
import torch

from .lightning_module import NNUE


class WeightClippingCallback(L.Callback):
    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch,
        batch_idx: int,
    ) -> None:
        assert isinstance(pl_module, NNUE)
        pl_module.model.clip_weights()


class CUDAGraphMarkStepCallback(L.Callback):
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def _mark_step_begin(self) -> None:
        if not self.enabled:
            return
        torch.compiler.cudagraph_mark_step_begin()

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch,
        batch_idx: int,
    ) -> None:
        _ = trainer
        _ = pl_module
        _ = batch
        _ = batch_idx
        self._mark_step_begin()

    def on_validation_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _ = trainer
        _ = pl_module
        _ = batch
        _ = batch_idx
        _ = dataloader_idx
        self._mark_step_begin()
