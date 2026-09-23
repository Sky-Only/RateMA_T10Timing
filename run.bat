@echo off
REM ============================================================
REM  RateMA_T10Timing - one-click runner
REM  Double-click this file to run the whole pipeline.
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
uv run ratema all
if errorlevel 1 goto err

echo.
echo [3/3] Done! Results are in the output\ folder:
echo.
echo     output\report.md      - report (read this first)
echo     output\charts\        - PNG charts
echo     output\summary.csv    - performance summary
echo     output\metrics.json   - full metrics
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
