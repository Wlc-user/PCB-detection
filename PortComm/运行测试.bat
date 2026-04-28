@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo.
echo ========================================
echo   PortAI 连接测试
echo ========================================
echo.

python 连接测试.py

pause
