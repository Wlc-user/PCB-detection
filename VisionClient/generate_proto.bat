@echo off
cd /d e:\pyspace\opencv\VisionClient
echo Generating proto files...
dotnet tool install --global Grpc.Tools 2>nul
protoc --csharp_out=. --grpc_out=. --plugin=protoc-gen-grpc=grpc_csharp_plugin.exe grpc_stream.proto
echo Done!
dir *.cs
