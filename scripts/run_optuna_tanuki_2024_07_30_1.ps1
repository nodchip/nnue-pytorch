$root = 'C:\shogi\training_data\tanuki-.nnue-pytorch-2024-07-30.1'

$allFiles = Get-ChildItem -Path $root -File |
  Where-Object { $_.Extension -eq '.bin' } |
  Sort-Object Name

$valFile = $allFiles |
  Where-Object { $_.Name -eq 'kifu.tag=train.depth=9.num_positions=1000000000.start_time=1724740898.thread_index=126.bin' } |
  Select-Object -ExpandProperty FullName

$trainFiles = $allFiles |
  Where-Object { $_.FullName -ne $valFile } |
  ForEach-Object FullName

$trainListFile = Join-Path $PSScriptRoot 'tanuki_2024_07_30_1_train_files.txt'
$valListFile = Join-Path $PSScriptRoot 'tanuki_2024_07_30_1_val_files.txt'

$trainFiles | Set-Content -Path $trainListFile -Encoding utf8
@($valFile) | Set-Content -Path $valListFile -Encoding utf8

Write-Host "train files: $($trainFiles.Count)"
Write-Host "val file:    $valFile"
Write-Host "train list:  $trainListFile"
Write-Host "val list:    $valListFile"

venv\Scripts\python.exe scripts\optuna_tune_train.py `
  --study-name tanuki-train-optuna-2024-07-30-1 `
  --storage sqlite:///tanuki_train_optuna.db `
  --trials 40 `
  --lr-min 6.5e-4 `
  --lr-max 1.05e-3 `
  --gamma-min 0.991 `
  --gamma-max 0.9945 `
  --beta1-min 0.88 `
  --beta1-max 0.93 `
  --beta2-min 0.997 `
  --beta2-max 0.9995 `
  -- `
  --features HalfKA_hm `
  --lambda 1.0 `
  --max_epochs 6 `
  --epoch-size 100000000 `
  --validation-size 1000000 `
  --num-workers 8 `
  --threads 8 `
  --compile-backend cudagraphs `
  --progress-log-interval 1000 `
  --network-save-period 1000000000 `
  --default_root_dir D:\hnoda\shogi\nnue-python.optuna `
  "@$trainListFile" `
  --validation-data "@$valListFile"
