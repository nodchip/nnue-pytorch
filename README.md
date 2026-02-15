# NNUE PyTorch

このリポジトリは、NNUE 学習および関連ツールを扱うための環境です。  
本READMEでは、環境構築、既存の学習導線、「進行度学習」関連スクリプトの使い方をまとめます。

## セットアップ

### Docker

ローカルに Python 環境や C++ ビルド環境を直接構築せず、Docker コンテナ上で実行できます。

#### 前提条件

AMD 環境:
- Docker
- 最新の ROCm ドライバ

NVIDIA 環境:
- Docker
- 最新の NVIDIA ドライバ
- NVIDIA Container Toolkit

ドライバ要件の詳細:
- [Running ROCm Docker containers (AMD)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html)
- [PyTorch container release notes (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-04.html#rel-25-04)

コンテナには CUDA 12.x / ROCm 6.4.3 と必要な依存関係が含まれます。  
ローカルの CUDA/ROCm ツールキットのバージョンは基本的に問いません。

### コンテナ起動

```bash
./run_docker.sh
```

起動時に GPU ベンダーとデータディレクトリを選択します。  
選択したデータディレクトリはコンテナにマウントされ、コンテナ内でそのまま学習コマンドを実行できます。

注意:
- イメージビルドには時間とディスク容量（目安 30-60GB）が必要です。

## 既存の学習フロー

- 詳細版: [wiki (train.py)](https://github.com/official-stockfish/nnue-pytorch/wiki/Basic-training-procedure-(train.py))
- 簡易版: [wiki (easy_train.py)](https://github.com/official-stockfish/nnue-pytorch/wiki/Basic-training-procedure-(easy_train.py))

## ログ確認

```bash
tensorboard --logdir=logs
```

ブラウザで `http://localhost:6006/` を開いて確認します。

## 自動対局でベストネットを選別

```bash
python run_games.py --concurrency 16 --stockfish_exe ./stockfish.master --c_chess_exe ./c-chess-cli --ordo_exe ./ordo --book_file_name ./noob_3moves.epd run96
```

`run96` 配下の `.ckpt` を `.nnue` に変換しつつ対局を回し、`ordo` で順位付けします。  
スクリプトはループ実行で新規チェックポイントを監視するため、空きコアがあれば学習と並行実行できます。

## 進行度学習スクリプト

### 概要

追加した主なスクリプト:
- `train_progress.py`: 進行度モデル学習（線形 + sigmoid、教師値は各対局で 0.0→1.0）
- `progress_tools.py`: 教師値生成と `progress.bin` 形式出力
- `convert_progress_ckpt.py`: 学習チェックポイント（`.pt`）から `progress.bin` へ変換
- `visualize_progress_buckets.py`: 学習済み重みを使って局面を進行度バケットに分けて画像化
- `progress_bucket_viz.py`: SFEN の盤面描画（漢字駒、持ち駒表示、後手駒180度回転）

関連するデータローダ拡張:
- `data_loader/stream.py`
- `data_loader/dataset.py`
- `data_loader/_native.py`
- `training_data_loader.cpp`

### 学習（例）

`C:\shogi\training_data\tanuki-.nnue-pytorch-2024-07-30.1` を使用する例です。

```powershell
venv\Scripts\python.exe train_progress.py `
  --train-data C:\shogi\training_data\tanuki-.nnue-pytorch-2024-07-30.1 `
  --epochs 10 `
  --device cuda `
  --max-train-files 256 `
  --batch-size 4096 `
  --lr 2e-4 `
  --log-every-games 100 `
  --checkpoint progress_model_e10_f256_cuda.pt `
  --export-progress-bin progress_e10_f256_cuda.bin
```

主な引数:
- `--epochs`: エポック数
- `--max-train-files`: 学習対象ファイル数（0 で全件）
- `--device`: `cuda` または `cpu`
- `--log-every-games`: 途中経過の表示間隔（対局数）

### チェックポイントから `progress.bin` へ変換

```powershell
venv\Scripts\python.exe convert_progress_ckpt.py `
  --checkpoint progress_model_e10_f256_cuda.pt `
  --output progress_e10_f256_cuda.bin
```

### 進行度バケット可視化

```powershell
venv\Scripts\python.exe visualize_progress_buckets.py `
  --data C:\shogi\training_data\tanuki-.nnue-pytorch-2024-07-30.1 `
  --progress-bin progress_e10_f256_cuda.bin `
  --output-dir progress_bucket_images_e10_f256_cuda `
  --bucket-count 8 `
  --samples-per-bucket 25 `
  --max-positions 300000
```

出力:
- `progress_bucket_images_e10_f256_cuda/bucket_0.png` 〜 `bucket_7.png`
- `progress_bucket_images_e10_f256_cuda/summary.json`

## 謝辞

- Sopel - 高速な疎データローダ
- connormcmonigle - https://github.com/connormcmonigle/seer-nnue
- syzygy - http://www.talkchess.com/forum3/viewtopic.php?f=7&t=75506
- https://github.com/DanielUranga/TensorFlowNNUE
- https://hxim.github.io/Stockfish-Evaluation-Guide/
- dkappe - Ranger 最適化手法の提案（https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer）
