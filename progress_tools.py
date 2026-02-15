from __future__ import annotations

import numpy as np
import torch


def build_linear_targets(length: int) -> torch.Tensor:
    """対局内のサンプル数から 0.0..1.0 の教師値を生成する。"""
    if length <= 0:
        raise ValueError("length must be positive")
    if length == 1:
        return torch.zeros(1, dtype=torch.float32)
    return torch.linspace(0.0, 1.0, steps=length, dtype=torch.float32)


def export_progress_weights(weights: torch.Tensor, output_path: str) -> None:
    """学習済み重みを tanuki_progress.cpp 互換の progress.bin に保存する。"""
    if weights.ndim != 2:
        raise ValueError("weights must be 2D [SQ_NB, fe_end]")
    array = weights.detach().cpu().to(torch.float64).contiguous().numpy()
    array.astype(np.float64, copy=False).tofile(output_path)
