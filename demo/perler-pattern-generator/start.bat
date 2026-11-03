@echo off
rem Perler Pattern Generator launcher (ASCII-only for cmd parser safety)
title Perler Pattern Generator
cd /d "%~dp0"

rem ==== Find a working Python (skip Microsoft Store alias) ====
rem PYEXE     = first python that runs
rem PYEXE_PIL = first python that runs AND has Pillow (preferred)
set "PYEXE="
set "PYEXE_PIL="

call :TRY "%USERPROFILE%\miniconda3\python.exe"
call :TRY "%USERPROFILE%\anaconda3\python.exe"
call :TRY "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
call :TRY "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
call :TRY "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
call :TRY "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
call :TRY "C:\Python313\python.exe"
call :TRY "C:\Python312\python.exe"
call :TRY "C:\Python311\python.exe"
call :TRY "C:\Python310\python.exe"
call :TRY "%USERPROFILE%\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"

for %%N in (python python3) do (
  for /f "delims=" %%F in ('where %%N 2^>nul') do (
    echo %%F | findstr /i "WindowsApps" >nul
    if errorlevel 1 call :TRY "%%F"
  )
)

if not defined PYEXE (
  where py >nul 2>nul && py -3 -c "import sys" >nul 2>nul && set "PYEXE=py -3"
)

if not defined PYEXE_PIL if not defined PYEXE (
  echo [ERROR] No working Python found.
  echo Install Python 3.10+ from https://www.python.org/downloads/
  echo and check "Add python.exe to PATH", then run this script again.
  pause
  exit /b 1
)

if defined PYEXE_PIL (
  set "PY=%PYEXE_PIL%"
  goto RUN
)

set "PY=%PYEXE%"
echo Using Python: %PY%  (Pillow missing, installing...)
%PY% -m pip install pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 %PY% -m pip install pillow
%PY% -c "import PIL" >nul 2>nul
if errorlevel 1 (
  echo.
  echo [ERROR] Pillow install failed, likely a network issue.
  echo Fix the network and run again, or install manually:
  echo     %PY% -m pip install pillow
  pause
  exit /b 1
)

:RUN
echo Using Python: %PY%
echo Starting local server: http://127.0.0.1:8765
%PY% server.py 8765
pause
exit /b 0

:TRY
if not exist %1 exit /b 0
%1 -c "import sys" >nul 2>nul || exit /b 0
if not defined PYEXE set "PYEXE=%~1"
%1 -c "import PIL" >nul 2>nul || exit /b 0
if not defined PYEXE_PIL set "PYEXE_PIL=%~1"
exit /b 0
