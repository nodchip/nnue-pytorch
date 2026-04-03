import argparse
import os
import re
import subprocess
import sys

import optuna


VAL_LOSS_PATTERN = re.compile(r"\bval_loss=([0-9]*\.?[0-9]+)\b")


def extract_val_loss(line: str) -> float | None:
    match = VAL_LOSS_PATTERN.search(line)
    if not match:
        return None
    return float(match.group(1))


def build_train_command(
    python_executable: str,
    train_script: str,
    base_args: list[str],
    trial_params: dict[str, float],
) -> list[str]:
    command = [python_executable, train_script, *base_args]
    command.extend(
        [
            "--progress-mode",
            "log",
            "--lr",
            f"{trial_params['lr']}",
            "--gamma",
            f"{trial_params['gamma']}",
            "--beta1",
            f"{trial_params['beta1']}",
            "--beta2",
            f"{trial_params['beta2']}",
        ]
    )
    return command


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune train.py hyperparameters with Optuna."
    )
    parser.add_argument("--study-name", type=str, default="train-py-optuna")
    parser.add_argument("--storage", type=str, default="")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--lr-min", type=float, default=5.0e-4)
    parser.add_argument("--lr-max", type=float, default=1.2e-3)
    parser.add_argument("--gamma-min", type=float, default=0.990)
    parser.add_argument("--gamma-max", type=float, default=0.997)
    parser.add_argument("--beta1-min", type=float, default=0.85)
    parser.add_argument("--beta1-max", type=float, default=0.95)
    parser.add_argument("--beta2-min", type=float, default=0.995)
    parser.add_argument("--beta2-max", type=float, default=0.9995)
    parser.add_argument(
        "--train-script",
        type=str,
        default=os.path.join(os.getcwd(), "train.py"),
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to train.py. Prefix them with --.",
    )
    return parser


def create_study(args: argparse.Namespace) -> optuna.Study:
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    if args.storage:
        return optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            direction="minimize",
            load_if_exists=True,
            pruner=pruner,
        )
    return optuna.create_study(direction="minimize", pruner=pruner)


def suggest_trial_params(
    trial: optuna.Trial, args: argparse.Namespace
) -> dict[str, float]:
    return {
        "lr": trial.suggest_float("lr", args.lr_min, args.lr_max, log=True),
        "gamma": trial.suggest_float("gamma", args.gamma_min, args.gamma_max),
        "beta1": trial.suggest_float("beta1", args.beta1_min, args.beta1_max),
        "beta2": trial.suggest_float("beta2", args.beta2_min, args.beta2_max),
    }


def run_trial(
    trial: optuna.Trial,
    args: argparse.Namespace,
    python_executable: str,
    train_script: str,
    base_args: list[str],
) -> float:
    trial_params = suggest_trial_params(trial, args)
    command = build_train_command(
        python_executable=python_executable,
        train_script=train_script,
        base_args=base_args,
        trial_params=trial_params,
    )
    print(f"[trial {trial.number}] {' '.join(command)}", flush=True)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    last_val_loss = None
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        value = extract_val_loss(line)
        if value is None:
            continue
        last_val_loss = value
        trial.report(value, step=trial.number)
        if trial.should_prune():
            process.terminate()
            raise optuna.TrialPruned()

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"train.py failed with exit code {return_code}")
    if last_val_loss is None:
        raise RuntimeError("No val_loss was found in train.py output")
    return last_val_loss


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    base_args = list(args.train_args)
    if base_args and base_args[0] == "--":
        base_args = base_args[1:]
    if not base_args:
        raise SystemExit("Pass train.py arguments after --")

    python_executable = sys.executable
    study = create_study(args)

    def objective(trial: optuna.Trial) -> float:
        return run_trial(
            trial=trial,
            args=args,
            python_executable=python_executable,
            train_script=args.train_script,
            base_args=base_args,
        )

    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout if args.timeout > 0 else None,
    )

    print(f"best_value={study.best_value}")
    print(f"best_params={study.best_params}")


if __name__ == "__main__":
    main()
