import argparse
import glob
import os
import time

import torch

import data_loader
from progress_tools import build_linear_targets, export_progress_weights


class ProgressModel(torch.nn.Module):
    """tanuki progress 推論式と同型の線形 + sigmoid モデル。"""

    def __init__(self, num_weights: int):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros(num_weights, dtype=torch.float32))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        gathered = self.weights[indices]
        return torch.sigmoid(gathered.sum(dim=1))


class TrainingProgressReporter:
    """一定対局数ごとに学習進捗を文字列化して表示する。"""

    def __init__(
        self,
        epoch: int,
        total_epochs: int,
        log_every_games: int,
        start_time: float | None = None,
        time_fn=None,
    ):
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.log_every_games = log_every_games
        self.time_fn = time_fn or time.time
        self.start_time = self.time_fn() if start_time is None else start_time

    def maybe_report(
        self,
        file_index: int,
        file_count: int,
        total_games: int,
        avg_loss: float,
        last_loss: float,
    ) -> str | None:
        if self.log_every_games <= 0:
            return None
        if total_games % self.log_every_games != 0:
            return None
        elapsed = self.time_fn() - self.start_time
        message = (
            f"progress epoch={self.epoch}/{self.total_epochs} "
            f"file={file_index}/{file_count} games={total_games} "
            f"avg_loss={avg_loss:.6f} loss={last_loss:.6f} elapsed={elapsed:.1f}s"
        )
        print(message, flush=True)
        return message


def iter_games_from_stream(
    filenames: list[str],
    batch_size: int,
    device: torch.device,
):
    """順序保持ストリームから対局単位でインデックス列を返す。"""
    dataset = data_loader.ProgressBatchDataset(
        filenames=filenames,
        batch_size=batch_size,
        cyclic=False,
        num_workers=1,
    )
    iterator = iter(dataset)
    current_indices: list[torch.Tensor] = []
    prev_ply: int | None = None
    num_weights: int | None = None
    while True:
        try:
            plies, indices, batch_num_weights = next(iterator)
        except StopIteration:
            break

        if num_weights is None and hasattr(iterator, "stream"):
            # dataset provider経由では属性に直接触れないため通常は使われない。
            pass
        plies = plies.tolist()
        for i, ply in enumerate(plies):
            if prev_ply is not None and ply < prev_ply:
                if current_indices:
                    yield torch.stack(current_indices, dim=0), num_weights
                current_indices = []
            current_indices.append(indices[i].to(device=device, non_blocking=True))
            prev_ply = ply
        if num_weights is None:
            num_weights = int(batch_num_weights)

    if current_indices:
        yield torch.stack(current_indices, dim=0), num_weights


def expand_inputs(paths: list[str]) -> list[str]:
    """入力パス群（ファイル・ディレクトリ・glob）を .bin 一覧へ展開する。"""
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


def train_one_epoch(
    model: ProgressModel,
    files: list[str],
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_games: int,
    epoch: int,
    total_epochs: int,
    log_every_games: int,
) -> float:
    """1エポック分の学習を実行して平均損失を返す。"""
    model.train()
    total_loss = 0.0
    total_games = 0
    reporter = TrainingProgressReporter(
        epoch=epoch,
        total_epochs=total_epochs,
        log_every_games=log_every_games,
    )
    for file_index, file_path in enumerate(files, start=1):
        for game_indices, _ in iter_games_from_stream([file_path], batch_size, device):
            targets = build_linear_targets(game_indices.shape[0]).to(device)
            preds = model(game_indices)
            loss = torch.nn.functional.mse_loss(preds, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.item())
            total_loss += loss_value
            total_games += 1
            reporter.maybe_report(
                file_index=file_index,
                file_count=len(files),
                total_games=total_games,
                avg_loss=total_loss / total_games,
                last_loss=loss_value,
            )
            if max_games > 0 and total_games >= max_games:
                break
        if max_games > 0 and total_games >= max_games:
            break
    if total_games == 0:
        raise RuntimeError("No games were found in input data")
    return total_loss / total_games


@torch.no_grad()
def evaluate(
    model: ProgressModel,
    files: list[str],
    batch_size: int,
    device: torch.device,
    max_games: int,
) -> float:
    """検証データの平均損失を計算する。"""
    model.eval()
    total_loss = 0.0
    total_games = 0
    for file_path in files:
        for game_indices, _ in iter_games_from_stream([file_path], batch_size, device):
            targets = build_linear_targets(game_indices.shape[0]).to(device)
            preds = model(game_indices)
            loss = torch.nn.functional.mse_loss(preds, targets)
            total_loss += float(loss.item())
            total_games += 1
            if max_games > 0 and total_games >= max_games:
                break
        if max_games > 0 and total_games >= max_games:
            break
    if total_games == 0:
        raise RuntimeError("No games were found in validation data")
    return total_loss / total_games


def main():
    parser = argparse.ArgumentParser(description="Train tanuki progress model with PyTorch.")
    parser.add_argument(
        "--train-data",
        action="append",
        nargs="+",
        required=True,
        help="Input .bin file, wildcard, or directory.",
    )
    parser.add_argument(
        "--val-data",
        action="append",
        nargs="+",
        default=None,
        help="Validation .bin file, wildcard, or directory.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-train-files", type=int, default=0)
    parser.add_argument("--max-val-files", type=int, default=0)
    parser.add_argument("--max-train-games", type=int, default=0)
    parser.add_argument("--max-val-games", type=int, default=0)
    parser.add_argument("--log-every-games", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default="progress_model.pt")
    parser.add_argument("--export-progress-bin", type=str, default="")
    args = parser.parse_args()

    train_inputs = sum(args.train_data, [])
    train_files = expand_inputs(train_inputs)
    if args.max_train_files > 0:
        train_files = train_files[: args.max_train_files]

    if args.val_data:
        val_inputs = sum(args.val_data, [])
        val_files = expand_inputs(val_inputs)
    else:
        split = max(1, len(train_files) // 20)
        val_files = train_files[:split]
        train_files = train_files[split:]
    if args.max_val_files > 0:
        val_files = val_files[: args.max_val_files]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"train_files={len(train_files)} val_files={len(val_files)} device={device}")

    probe_dataset = data_loader.ProgressBatchDataset(
        filenames=[train_files[0]],
        batch_size=32,
        cyclic=False,
        num_workers=1,
    )
    probe_iter = iter(probe_dataset)
    _, _, num_weights = next(probe_iter)
    model = ProgressModel(num_weights=num_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            files=train_files,
            batch_size=args.batch_size,
            optimizer=optimizer,
            device=device,
            max_games=args.max_train_games,
            epoch=epoch,
            total_epochs=args.epochs,
            log_every_games=args.log_every_games,
        )
        val_loss = evaluate(
            model=model,
            files=val_files,
            batch_size=args.batch_size,
            device=device,
            max_games=args.max_val_games,
        )
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_weights": num_weights,
                    "best_val_loss": best_val,
                },
                args.checkpoint,
            )
            print(f"saved checkpoint: {args.checkpoint}")

    if args.export_progress_bin:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        weights = model.weights.detach().cpu().view(81, -1)
        export_progress_weights(weights, args.export_progress_bin)
        print(f"exported: {args.export_progress_bin}")


if __name__ == "__main__":
    main()
