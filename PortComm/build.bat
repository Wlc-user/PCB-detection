@echo off
cd /d e:\pyspace\opencv\PortComm
echo Building PortComm.csproj...
dotnet build PortComm.csproj -c Release > build.log 2>&1
type build.log
echo.
if exist "bin\Release\net9.0-windows\PortComm.exe" (
    echo SUCCESS: EXE created!
    dir "bin\Release\net9.0-windows\PortComm.exe"
) else (
    echo FAILED: No EXE found
)
pause
