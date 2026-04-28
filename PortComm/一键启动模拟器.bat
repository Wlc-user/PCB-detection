@echo off
chcp 65001 >nul
echo ========================================
echo   PortAI 一键启动 (本地模拟器模式)
echo ========================================
echo.

cd /d "%~dp0"

echo [1/2] 启动PLC模拟器...
start "PLC_Simulator" cmd /k "python start_plc.py"

timeout /t 2 /nobreak >nul

echo [2/2] 启动HMI上位机...
start "PortAI_HMI" cmd /k "python production_hmi.py"

echo.
echo ========================================
echo   启动完成！
echo ========================================
echo.
echo PLC模拟器窗口: 输入 1 选择TCP模式
echo HMI窗口: 点击"连接"按钮
echo.
pause
