@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo.
echo ========================================
echo   PortAI 高级HMI (状态机版)
echo ========================================
echo.

echo Starting PLC simulator...
start "PLC" cmd /k "python start_plc.py"

timeout /t 2 /nobreak >nul

echo Starting Advanced HMI...
start "HMI" cmd /k "python advanced_hmi.py"

echo.
echo Started! Check the windows.
pause
