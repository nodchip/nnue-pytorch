import struct

import torch

from progress_tools import build_linear_targets, export_progress_weights


def test_build_linear_targets():
    targets = build_linear_targets(5)
    expected = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32)
    assert torch.allclose(targets, expected)


def test_export_progress_weights(tmp_path):
    path = tmp_path / "progress.bin"
    weights = torch.tensor([[0.5, -1.25], [2.0, 3.5]], dtype=torch.float32)
    export_progress_weights(weights, str(path))

    data = path.read_bytes()
    assert len(data) == 4 * 8
    values = struct.unpack("<4d", data)
    assert values == (0.5, -1.25, 2.0, 3.5)
