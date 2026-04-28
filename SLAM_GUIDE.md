# 无人物流感知 + SLAM 系统

## 概述

完整的无人物流小车感知系统，集成：
- **视觉感知**: 目标检测、分割、追踪
- **SLAM定位**: 视觉SLAM、激光SLAM、融合SLAM
- **统一接口**: REST API、C++集成

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    无人物流感知 + SLAM 系统                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐    │
│   │                    输入数据                               │    │
│   │   摄像头 (RGB/D)          激光雷达          里程计        │    │
│   └─────────────────────────┬──────────────────────────────┘    │
│                             │                                    │
│   ┌─────────────────────────┴──────────────────────────────┐    │
│   │              统一感知 + SLAM 引擎                         │    │
│   │                                                         │    │
│   │   ┌─────────────┐    ┌─────────────┐    ┌────────────┐ │    │
│   │   │  视觉感知   │    │   SLAM      │    │   融合    │ │    │
│   │   │  (YOLOv8)  │◄──►│ (V/L/F)    │◄──►│ 感知+SLAM │ │    │
│   │   └─────────────┘    └─────────────┘    └────────────┘ │    │
│   │          │                  │                  │        │    │
│   │          │                  │                  │        │    │
│   │          └──────────────────┴──────────────────┘        │    │
│   │                           │                            │    │
│   │                           ▼                            │    │
│   │   ┌─────────────────────────────────────────────────┐  │    │
│   │   │              导航决策输出                        │  │    │
│   │   │  位姿 | 障碍物 | 可通行区域 | 危险区域 | 地图     │  │    │
│   │   └─────────────────────────────────────────────────┘  │    │
│   └─────────────────────────────────────────────────────────┘    │
│                             │                                    │
│                             ▼                                    │
│   ┌─────────────────────────────────────────────────────────┐    │
│   │                    API 服务层                            │    │
│   │    REST API (JSON)     |     gRPC (Protocol Buffers)    │    │
│   └─────────────────────────────────────────────────────────┘    │
│                             │                                    │
│                             ▼                                    │
│   ┌─────────────────────────────────────────────────────────┐    │
│   │                   C++ 嵌入式客户端                       │    │
│   │              AGV控制器 | 导航系统                        │    │
│   └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 模块说明

### 1. 视觉感知 (`logistics_perception.py`)

| 功能 | 说明 |
|------|------|
| 目标检测 | YOLOv8 (80+ 类别) |
| 类别映射 | COCO → 物流类别 (person, vehicle, box, obstacle) |
| 场景理解 | 仓库/无人车/混合场景识别 |

### 2. SLAM 模块 (`slam_module.py`)

| SLAM类型 | 说明 | 适用场景 |
|----------|------|----------|
| `VisualSLAM` | 纯视觉SLAM (ORB-SLAM风格) | 有摄像头的场景 |
| `LidarSLAM` | 激光SLAM (GMapping风格) | 有激光雷达的场景 |
| `FusionSLAM` | 视觉+激光融合 (LIO-SAM风格) | 同时有相机和激光 |

### 3. 统一系统 (`unified_perception_slam.py`)

融合视觉感知和SLAM，提供完整的环境理解：

```python
from unified_perception_slam import create_unified_system

# 创建融合系统
system = create_unified_system('fusion')

# 更新
result = system.update(frame=frame, scan=scan)

# 结果
print(f"位姿: ({result.pose.x}, {result.pose.y}, {result.pose.theta})")
print(f"障碍物: {len(result.obstacles)}")
print(f"路径通畅: {result.path_clear}")
```

## API 服务

### 启动服务

```bash
# REST API (推荐快速开发)
python unified_server.py

# gRPC (推荐生产)
python grpc_perception_server.py
```

### REST API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `POST /api/detect` | POST | 目标检测 |
| `POST /api/fusion` | POST | 融合感知 (图像+激光) |
| `POST /api/navigate` | POST | 导航决策 |
| `GET /api/pose` | GET | 获取位姿 |
| `GET /api/map` | GET | 获取地图 |
| `GET /api/status` | GET | 系统状态 |

### C++ 集成示例

```cpp
#include <curl/curl.h>
#include <opencv2/opencv.hpp>

// 融合感知请求
struct FusionRequest {
    std::string image_base64;
    std::vector<float> lidar_scan;  // 360度激光数据
    std::string camera_id;
};

// 发送融合感知请求
nlohmann::json fusion_detect(cv::Mat& frame, const std::vector<float>& scan) {
    // 编码图像
    std::vector<unsigned char> buf;
    cv::imencode(".jpg", frame, buf);
    std::string img_base64 = base64_encode(buf.data(), buf.size());
    
    // 构建请求
    nlohmann::json req = {
        {"image_base64", img_base64},
        {"lidar_scan", scan},
        {"camera_id", "agv_front"}
    };
    
    // 发送
    auto resp = curl_post("http://localhost:8080/api/fusion", req.dump());
    
    return nlohmann::json::parse(resp);
}

// 处理响应
void handle_response(const nlohmann::json& resp) {
    // 位姿
    auto pose = resp["pose"];
    printf("Pose: (%.2f, %.2f, %.2f)\n", 
           pose["x"], pose["y"], pose["theta"]);
    
    // 障碍物
    for (auto& obs : resp["obstacles"]) {
        if (obs["dangerous"]) {
            // 危险障碍物 - 停车
            send_command({"cmd": "stop", "speed": 0});
        }
    }
    
    // 路径判断
    if (resp["path_clear"]) {
        send_command({"cmd": "move_forward", "speed": 0.5});
    }
}
```

## 导航决策

系统自动分析前方区域，返回导航建议：

```python
{
    "obstacles": [
        {"x": 1.5, "y": 0.2, "category": "person", "dangerous": true},
        {"x": 3.0, "y": -0.5, "category": "box", "dangerous": false}
    ],
    "can_proceed": false,
    "action": "stop"  # 或 "move_forward", "turn_left", "turn_right"
}
```

## 地图输出

支持输出占据栅格地图：

```python
{
    "type": "occupancy_grid",
    "width": 200,
    "height": 200,
    "resolution": 0.05,  # 5cm/pixel
    "origin": (-5.0, -5.0),
    "data": "..."  # base64编码的栅格数据
}
```

## 依赖

```
# 核心依赖
numpy>=1.21
opencv-python>=4.5
ultralytics>=8.0  # YOLOv8

# 可选依赖
torch>=2.0        # GPU加速
torchvision>=0.15
fastapi>=0.100    # REST API
uvicorn>=0.23     # ASGI服务器
grpcio>=1.50      # gRPC
```

## 文件结构

```
.
├── logistics_perception.py      # 视觉感知模块
├── slam_module.py                # SLAM模块
├── unified_perception_slam.py    # 统一感知+SLAM
├── unified_server.py             # REST API服务
├── grpc_perception_server.py     # gRPC服务
├── cpp/
│   ├── CMakeLists.txt
│   ├── fusion_client.cpp         # 融合感知客户端
│   └── rest_client.cpp          # REST客户端
├── INTEGRATION_GUIDE.md          # 集成指南
└── SLAM_GUIDE.md                 # 本文档
```

## 使用场景

### 仓库AGV

```python
# 只用摄像头 + 视觉SLAM
system = create_unified_system('visual')
```

### 室外无人车

```python
# 摄像头 + 激光 + 融合
system = create_unified_system('fusion')
```

### 简单场景

```python
# 只用感知，无SLAM
simple = SimpleAGVPerception()
result = simple.update(frame)
```

## 许可证

MIT License
