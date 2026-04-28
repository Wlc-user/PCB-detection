@echo off
chcp 65001 >nul
title PortAI 工业通信系统

cd /d "%~dp0"

echo.
echo ========================================
echo   PortAI 工业通信系统 - 一键启动
echo ========================================
echo.

REM 关闭已有的Python进程
taskkill /F /IM python.exe 2>nul
timeout /t 1 /nobreak >nul

echo [1/3] 启动PLC模拟器...
start "PLC模拟器" cmd /k "python start_plc.py"

echo 等待PLC启动...
timeout /t 2 /nobreak >nul

REM 检查端口
netstat -ano | findstr ":5000" | findstr "LISTENING" >nul
if %errorlevel% neq 0 (
    echo [警告] 端口5000未就绪，尝试继续...
)

echo [2/3] PLC模拟器已启动 (127.0.0.1:5000)
echo [3/3] 启动HMI上位机...
start "PortAI上位机" cmd /k "python production_hmi.py"

echo.
echo ========================================
echo   启动完成！
echo ========================================
echo.
echo PLC模拟器窗口会显示 Tick 和 Regs 数据
echo HMI窗口点击"连接"按钮即可
echo.
