@echo off
setlocal enabledelayedexpansion

REM One-click Windows environment repair for UReader-main (ROLLBACK VERSION - before pyarrow fix)
REM Usage: scripts\fix_env_win.bat

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
  echo [ERROR] conda not found in PATH.
  exit /b 1
)

echo [2/7] Activating conda env: %ENV_NAME%
call conda activate %ENV_NAME%
if errorlevel 1 (
  echo [ERROR] Failed to activate conda env.
  exit /b 1
)

echo [3/7] Cleaning conflicting packages...
python -m pip uninstall -y torch torchvision torchaudio numpy datasets requests urllib3 chardet charset-normalizer >nul 2>nul
call conda remove -y pytorch torchvision torchaudio pytorch-cuda >nul 2>nul

echo [4/7] Installing PyTorch 1.13.1 + CUDA 11.7...
call conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
if errorlevel 1 (
  echo [ERROR] Failed to install PyTorch stack.
  exit /b 1
)

echo [5/7] Installing project dependencies...
python -m pip install --upgrade pip
python -m pip install --no-deps -r requirement_win.txt
if errorlevel 1 (
  echo [ERROR] Failed to install requirement_win.txt
  exit /b 1
)


echo [6/7] Runtime diagnostics...
python -c "import torch, torchvision, numpy, datasets; import torchvision.ops as ops; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('numpy', numpy.__version__); print('cuda runtime', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print('nms ok', hasattr(ops, 'nms'))"
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