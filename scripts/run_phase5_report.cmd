@echo off
setlocal

REM --- EDIT THIS IF YOUR REPO IS IN A DIFFERENT FOLDER ---
set "REPO=D:\Ticker Tattle"

set "PHASE4=%REPO%\artifacts\phase4"
set "PHASE3=%REPO%\artifacts\phase3"
set "OUT=%REPO%\artifacts\phase5"

REM Ensure paths exist
if not exist "%OUT%" mkdir "%OUT%"

REM Optional: route Phase-* writers to this repo (used by our scripts)
set "NBSI_OUT_ROOT=%REPO%"

REM Run Phase-5 report (no network; reads Phase-4/Phase-3 artifacts)
cd /d "%REPO%"

REM Rotate Phase-5 QA log before each run (keeps .1..7)
set "LOG=%OUT%\qa_phase5.log"
if exist "%REPO%\scripts\rotate_log.cmd" call "%REPO%\scripts\rotate_log.cmd" "%LOG%"

python nbsi\phase5\scripts\run_phase5.py --from "%PHASE4%" --out "%OUT%" --phase3-root "%PHASE3%"

exit /b %ERRORLEVEL%

