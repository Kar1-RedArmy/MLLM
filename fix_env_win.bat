@echo off
setlocal enabledelayedexpansion

REM One-click Windows environment repair for UReader-main
REM Usage:
REM   scripts\fix_env_win.bat            (uses env name MLLM)
REM   scripts\fix_env_win.bat myenv      (uses custom conda env)

set ENV_NAME=MLLM
if not "%~1"=="" set ENV_NAME=%~1

echo [1/7] Enter repo root...
cd /d "%~dp0.."
if not exist "pipeline\train.py" (
  echo [ERROR] Please run this script inside UReader-main\scripts.
  exit /b 1
)

where conda >nul 2>nul
if errorlevel 1 (
  echo [ERROR] conda not found in PATH. Please open Anaconda Prompt.
  exit /b 1
)

echo [2/7] Activating conda env: %ENV_NAME%
call conda activate %ENV_NAME%
if errorlevel 1 (
  echo [ERROR] Failed to activate conda env %ENV_NAME%.
  exit /b 1
)

echo [3/7] Cleaning conflicting torch installs...
python -m pip uninstall -y torch torchvision torchaudio >nul 2>nul
call conda remove -y pytorch torchvision torchaudio pytorch-cuda >nul 2>nul

echo [4/7] Installing matched torch/torchvision/torchaudio (CUDA 12.1 runtime)...
call conda install -y pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
if errorlevel 1 (
  echo [ERROR] Failed to install PyTorch stack.
  exit /b 1
)

echo [5/7] Installing project dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirement_win.txt
python -m pip install datasets  REM Install missing Hugging Face datasets library
python -m pip install requests --upgrade  REM Upgrade requests to fix dependency warnings
python -m pip install urllib3<2.0 chardet<6.0 charset-normalizer  REM Pin compatible dependencies
if errorlevel 1 (
  echo [ERROR] Failed to install requirement_win.txt, datasets, or requests fixes
  exit /b 1
)

echo [6/7] Runtime diagnostics...
python -c "import torch, torchvision; import torchvision.ops as ops; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('cuda runtime', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print('nms ok', hasattr(ops, 'nms'))"
if errorlevel 1 (
  echo [ERROR] Torch/Torchvision runtime check failed.
  exit /b 1
)

echo [7/7] Train entry smoke test...
python -m pipeline.train --help >nul
if errorlevel 1 (
  echo [ERROR] Training entry still fails. Please share full traceback.
  exit /b 1
)

echo [DONE] Environment repaired and train entry check passed.
exit /b 0
