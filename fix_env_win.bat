@echo off
setlocal

REM Wrapper entry to avoid running stale script copies.
REM Always dispatch to the canonical script under UReader-main\scripts.

set SCRIPT_PATH=%~dp0UReader-main\scripts\fix_env_win.bat
if not exist "%SCRIPT_PATH%" (
  echo [ERROR] Canonical script not found: %SCRIPT_PATH%
  exit /b 1
)

echo [INFO] Delegating to %SCRIPT_PATH%
call "%SCRIPT_PATH%" %*
exit /b %errorlevel%
