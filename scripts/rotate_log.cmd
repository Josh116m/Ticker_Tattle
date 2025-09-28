@echo off
setlocal EnableDelayedExpansion
set "LOG=%~1"
if "%LOG%"=="" goto :eof

REM Safe rotation: LOG -> LOG.1 ... LOG.7 (delete oldest), suppress errors/noise
for /L %%i in (7,-1,2) do (
  set /a j=%%i-1
  if exist "%LOG%.!j!" (
    if exist "%LOG%.%%i" del /q "%LOG%.%%i" >nul 2>&1
    ren "%LOG%.!j!" "%~nx1.%%i" >nul 2>&1
  )
)
if exist "%LOG%" (
  if exist "%LOG%.1" del /q "%LOG%.1" >nul 2>&1
  ren "%LOG%" "%~nx1.1" >nul 2>&1
)

endlocal
