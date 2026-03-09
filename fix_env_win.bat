@echo off
setlocal

REM Root wrapper: always dispatch to UReader-main/scripts/fix_env_win.bat
set TARGET=%~dp0UReader-main\scripts\fix_env_win.bat
if not exist "%TARGET%" (
  echo [ERROR] Cannot find target script: %TARGET%
  echo [HINT] Please ensure repo layout contains UReader-main\scripts\fix_env_win.bat
  exit /b 1
)

echo [INFO] Wrapper script: %~f0
echo [INFO] Dispatch target: %TARGET%

if "%~1"=="" (
  call "%TARGET%"
) else (
  call "%TARGET%" %1
)
exit /b %errorlevel%
