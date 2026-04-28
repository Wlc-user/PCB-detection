@echo off
cd /d e:\pyspace\opencv\VisionClient
dotnet build > build.log 2>&1
type build.log
