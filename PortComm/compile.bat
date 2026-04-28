@echo off
chcp 65001 >nul
echo ========================================
echo   PortAI C# 编译脚本
echo ========================================
echo.

:: 检查 .NET SDK
where dotnet >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 .NET SDK
    echo 请先安装: https://dotnet.microsoft.com/download
    echo.
    pause
    exit /b 1
)

:: 编译
echo [编译] COM_PROTOCOL.cs
csc /target:winexe /out:PortComm.exe /reference:System.Windows.Forms.dll /reference:System.Drawing.dll /reference:System.dll COM_PROTOCOL.cs

if %errorlevel% equ 0 (
    echo.
    echo [成功] 编译完成!
    echo 运行: PortComm.exe
    echo.
) else (
    echo.
    echo [失败] 编译出错
    echo.
)

pause
