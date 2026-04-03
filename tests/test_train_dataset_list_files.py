from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train


def test_expand_path_list_args_replaces_at_file_with_lines(tmp_path):
    list_file = tmp_path / "datasets.txt"
    list_file.write_text("train_a.bin\n\ntrain_b.bin\n", encoding="utf-8")

    expanded = train.expand_path_list_args(
        [f"@{list_file}", "train_c.bin"]
    )

    assert expanded == ["train_a.bin", "train_b.bin", "train_c.bin"]


def test_expand_path_list_args_leaves_regular_paths_untouched():
    expanded = train.expand_path_list_args(["train_a.bin", "train_b.bin"])

    assert expanded == ["train_a.bin", "train_b.bin"]
