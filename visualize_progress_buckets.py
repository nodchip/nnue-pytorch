import argparse
import glob
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch

import data_loader
from progress_bucket_viz import make_grid_figure, progress_to_bucket


def expand_inputs(paths: list[str]) -> list[str]:
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.bin"))))
        else:
            files.extend(sorted(glob.glob(p)))
    files = [f for f in files if f.lower().endswith(".bin")]
    if not files:
        raise FileNotFoundError("No .bin files found from inputs")
    return files


def load_progress_weights(progress_path: str, expected_num_weights: int) -> torch.Tensor:
    raw = np.fromfile(progress_path, dtype=np.float64)
    if raw.size != expected_num_weights:
        raise ValueError(
            f"progress size mismatch: expected={expected_num_weights} actual={raw.size}"
        )
    return torch.from_numpy(raw.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(
        description="Sample positions and visualize by progress bucket."
    )
    parser.add_argument(
        "--data",
        action="append",
        nargs="+",
        required=True,
        help="Input .bin file, wildcard, or directory.",
    )
    parser.add_argument(
        "--progress-bin",
        default="progress.bin",
        help="Progress weights file path.",
    )
    parser.add_argument("--output-dir", default="progress_bucket_images")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--bucket-count", type=int, default=8)
    parser.add_argument("--samples-per-bucket", type=int, default=25)
    parser.add_argument("--max-positions", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    inputs = sum(args.data, [])
    files = expand_inputs(inputs)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = data_loader.ProgressSfenBatchDataset(
        filenames=files,
        batch_size=args.batch_size,
        cyclic=False,
        num_workers=args.num_workers,
        config=data_loader.DataloaderSkipConfig(),
    )
    iterator = iter(dataset)

    plies, indices, sfens, num_weights = next(iterator)
    weights = load_progress_weights(args.progress_bin, num_weights)

    rng = random.Random(args.seed)
    seen = defaultdict(int)
    samples: dict[int, list[tuple[str, float]]] = {
        i: [] for i in range(args.bucket_count)
    }

    processed = 0
    while True:
        batch_progress = torch.sigmoid(weights[indices].sum(dim=1))
        for i in range(batch_progress.shape[0]):
            p = float(batch_progress[i].item())
            b = progress_to_bucket(p, bucket_count=args.bucket_count)
            seen[b] += 1
            if len(samples[b]) < args.samples_per_bucket:
                samples[b].append((sfens[i], p))
            else:
                pick = rng.randint(0, seen[b] - 1)
                if pick < args.samples_per_bucket:
                    samples[b][pick] = (sfens[i], p)

        processed += batch_progress.shape[0]
        if args.max_positions > 0 and processed >= args.max_positions:
            break
        try:
            plies, indices, sfens, _ = next(iterator)
        except StopIteration:
            break

    summary = {
        "processed_positions": processed,
        "bucket_count": args.bucket_count,
        "samples_per_bucket": args.samples_per_bucket,
        "seen_per_bucket": {str(k): int(v) for k, v in sorted(seen.items())},
        "images": {},
    }
    for b in range(args.bucket_count):
        out_path = os.path.join(args.output_dir, f"bucket_{b}.png")
        make_grid_figure(
            items=samples[b],
            output_path=out_path,
            cols=5,
            title=f"Bucket {b} ({len(samples[b])} samples)",
        )
        summary["images"][f"bucket_{b}"] = out_path
        print(f"saved: {out_path}")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
