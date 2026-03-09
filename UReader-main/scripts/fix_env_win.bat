@echo off
setlocal enabledelayedexpansion

REM One-click Windows environment repair for UReader-main
REM Usage:
REM   scripts\fix_env_win.bat            (uses env name MLLM)
REM   scripts\fix_env_win.bat myenv      (uses custom conda env)

set ENV_NAME=MLLM
if not "%~1"=="" set ENV_NAME=%~1

echo [1/7] Enter repo root...
set ROOT=%~dp0..
cd /d "%ROOT%"
echo [INFO] Repo root: %CD%
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

set REQ_FILE=requirement_win.txt
if exist "requiremnet_win.txt" set REQ_FILE=requiremnet_win.txt
if not exist "%REQ_FILE%" (
  echo [ERROR] Cannot find requirement file in repo root.
  echo [ERROR] Expected one of: requirement_win.txt or requiremnet_win.txt
  dir /b *.txt
  exit /b 1
)

echo [INFO] Using requirement file: %REQ_FILE%
python -m pip install -r %REQ_FILE%
if errorlevel 1 (
  echo [ERROR] Failed to install %REQ_FILE%
  exit /b 1
)

echo [INFO] Forcing PEFT/Transformers/Chardet compatible versions...
python -m pip uninstall -y peft transformers chardet >nul 2>nul
python -m pip install --no-cache-dir transformers==4.29.1 peft==0.3.0 "chardet<6"
if errorlevel 1 (
  echo [ERROR] Failed to force compatible peft/transformers/chardet versions.
  exit /b 1
)

set PYTHONPATH=%CD%
echo [INFO] PYTHONPATH set to: %PYTHONPATH%

echo [6/7] Runtime diagnostics...
python -c "import os, pipeline; print('cwd', os.getcwd()); print('pipeline path', pipeline.__file__)"
python -c "import torch, torchvision, transformers, peft, chardet; import torchvision.ops as ops; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('transformers', transformers.__version__); print('peft', peft.__version__); print('chardet', chardet.__version__); print('cuda runtime', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print('nms ok', hasattr(ops, 'nms'))"
if errorlevel 1 (
  echo [ERROR] Runtime diagnostic failed.
  exit /b 1
)

echo [7/7] Train entry smoke test...
python "%CD%\pipeline\train.py" --help >nul
if errorlevel 1 (
  echo [ERROR] Training entry still fails. Please share full traceback.
  exit /b 1
)

echo [DONE] Environment repaired and train entry check passed.
exit /b 0
