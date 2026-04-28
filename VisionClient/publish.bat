@echo off
cd /d e:\pyspace\opencv\VisionClient
echo Publishing...
dotnet publish -c Release -r win-x64 --self-contained false -o publish
echo Done!
dir publish\*.exe
