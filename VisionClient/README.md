# WPF Vision Client

## 项目说明
基于 WPF 的视觉检测客户端，支持 gRPC 和 WebSocket 两种协议连接视觉服务端。

## 功能特性
- ✅ 实时图像流显示
- ✅ gRPC 高性能调用
- ✅ WebSocket 实时推送
- ✅ 检测结果叠加显示
- ✅ 性能统计面板
- ✅ 多协议自动切换

## 快速开始

### 1. 安装依赖
```bash
cd VisionClient
dotnet restore
```

### 2. 配置服务器地址
编辑 `App.config` 中的服务器地址：
```xml
<add key="GrpcServer" value="localhost:50051"/>
<add key="WsServer" value="ws://localhost:8765"/>
```

### 3. 运行
```bash
dotnet run
```

## 界面预览
- 左侧：实时视频流/图像显示
- 右上：检测结果列表
- 右下：性能统计 (FPS, 延迟)

## 快捷键
| 快捷键 | 功能 |
|-------|------|
| Ctrl+O | 打开本地图像 |
| Ctrl+S | 打开本地视频 |
| Ctrl+C | 连接服务器 |
| Space | 截图保存 |
