@echo off
chcp 65001 >nul
title PortAI 启动器

echo.
echo ========================================
echo   PortAI 工业通信系统 - 一键启动
echo ========================================
echo.

cd /d "%~dp0"

REM 检查端口5000是否被占用
netstat -ano | findstr ":5000" >nul
if %errorlevel%==0 (
    echo [警告] 端口5000已被占用!
    echo.
    echo 请先关闭占用端口的程序，或在任务管理器中结束python进程
    echo.
    pause
    exit /b 1
)

echo [1/3] 启动PLC模拟器...
start "PLC_Simulator" cmd /k "echo 选择通信模式: && echo 1 - TCP && echo 2 - Serial && set /p choice=请输入选择: && if !choice!==1 (python start_plc.py) else (python start_plc.py)"

REM 等待PLC启动
echo 等待PLC启动...
timeout /t 3 /nobreak >nul

REM 检查PLC是否启动
netstat -ano | findstr ":5000" >nul
if %errorlevel% neq 0 (
    echo [错误] PLC模拟器启动失败!
    echo.
    pause
    exit /b 1
)

echo [2/3] PLC已启动 (127.0.0.1:5000)
echo [3/3] 启动HMI上位机...
start "PortAI_HMI" cmd /k "python production_hmi.py"

echo.
echo ========================================
echo   启动完成！
echo ========================================
echo.
echo PLC模拟器窗口: 输入 1 选择TCP模式
echo HMI窗口: 点击"连接"按钮
echo.
echo 按任意键退出此窗口...
pause >nul
