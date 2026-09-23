@echo off
REM ============================================================
REM  RateMA_T10Timing - one-click runner
REM  Double-click this file to run the whole pipeline.
REM
REM  All tunable parameters (windows, tolerance, direction,
REM  costs, output folder name, which steps to run ...) live in
REM  run.py - edit that file, then double-click this one.
REM
REM  NOTE: keep this file ASCII-only. cmd.exe reads .bat files
REM  using the OEM codepage (GBK on zh-CN Windows), so UTF-8
REM  Chinese text here would be mangled into broken commands.
REM  All Chinese output comes from the Python CLI instead.
REM ============================================================
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 goto nouv

echo.
echo [1/3] Installing / checking dependencies ...
uv sync
if errorlevel 1 goto err

echo.
echo [2/3] Running: Excel -^> CSV -^> backtest -^> sweep -^> charts ...
REM Runs the configurable entry point (all parameters live in run.py).
REM Any argument is passed through as the run name, e.g.  run.bat my_plan_A
uv run python run.py %*
if errorlevel 1 goto err

echo.
echo [3/3] Done! The exact output paths are printed just above.
echo.
echo     Results live in:  output\^<run_name^>\
echo.
echo     report.md            - report (read this first)
echo     composite_report.md  - equal-weight composite
echo     journal\journal.md   - daily trade journal
echo     charts\              - PNG charts
echo     summary.csv          - performance summary
echo     metrics.json         - full metrics
echo.
echo Opening the output folder ...
start "" "output"
echo.
pause
exit /b 0

:nouv
echo.
echo *** uv not found ***
echo.
echo Install uv first. Open PowerShell and run:
echo.
echo   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 ^| iex"
echo.
echo Then close this window, open a new one, and double-click run.bat again.
echo.
pause
exit /b 1

:err
echo.
echo *** FAILED. Please copy the error message above and send it to me. ***
echo.
pause
exit /b 1
