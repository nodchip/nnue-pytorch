import argparse

import torch

from progress_tools import export_progress_weights


def main():
    """学習チェックポイントから progress.bin を生成する。"""
    parser = argparse.ArgumentParser(
        description="Convert progress training checkpoint to tanuki progress.bin format."
    )
    parser.add_argument("--checkpoint", required=True, help="Input .pt checkpoint path.")
    parser.add_argument("--output", required=True, help="Output progress.bin path.")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    weights = state.get("weights")
    if weights is None:
        raise KeyError("weights not found in checkpoint")
    if weights.numel() % 81 != 0:
        raise ValueError("weights size is not divisible by SQ_NB(81)")
    reshaped = weights.view(81, -1)
    export_progress_weights(reshaped, args.output)
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
