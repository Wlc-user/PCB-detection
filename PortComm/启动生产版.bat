@echo off
chcp 65001 >nul
title PortAI 工业监控系统
cd /d %~dp0
echo.
echo ====================================
echo    PortAI 工业监控系统 v3.0
echo ====================================
echo.
echo 正在启动...
python production_hmi.py
echo.
echo 程序已退出
pause
