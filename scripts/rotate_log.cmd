@echo off
setlocal
set "LOG=%~1"
if "%LOG%"=="" goto :eof

REM Rotate qa_phase5.log -> qa_phase5.log.1 ... qa_phase5.log.7 (oldest dropped)
for /L %%i in (7,-1,2) do (
  if exist "%LOG%.%%i-1" ren "%LOG%.%%i-1" "%~nx1.%%i"
)
if exist "%LOG%" ren "%LOG%" "%~nx1.1"

endlocal

