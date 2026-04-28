# 无人物流感知系统

Logistics Perception System - 分割、检测、分类、追踪一体化解决方案

## 功能特性

### 四大核心能力

| 能力 | 技术 | 说明 |
|-----|------|------|
| **分割 (Segmentation)** | SAM / SAM2 | 精确物体边界分割 |
| **检测 (Detection)** | YOLOv8 / YOLOv10 | 80+ 目标检测 |
| **分类 (Classification)** | 自定义映射 | 物流专用类别 |
| **追踪 (Tracking)** | ByteTrack | 多目标实时追踪 |

### 场景支持

#### 仓库环境
- 人员检测与安全预警
- 叉车 AGV 追踪
- 货物/托盘计数
- 货架状态监控
- 传送带物体检测

#### 无人车环境
- 行人检测与避障
- 车辆/障碍物识别
- 交通标识检测
- 车道线检测
- 危险区域预警

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements_logistics.txt

# 可选: 安装 SAM 分割模型
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 2. 启动服务

```bash
# 启动感知服务
python logistics_server.py
```

### 3. 打开浏览器

```
http://localhost:8000/demo
```

## API 接口

### 流管理

```bash
# 创建流
POST /api/streams
{
    "stream_type": "camera",  # camera / rtsp / http
    "source": "0"             # 0=摄像头, rtsp://... , http://...
}

# 启动流
POST /api/streams/{id}/start

# 获取帧
GET /api/streams/{id}/frame

# 获取感知结果
GET /api/streams/{id}/perception
```

### 图像分析

```bash
# 分析图像
POST /api/perception/image
Content-Type: image/*

# 返回检测结果
{
    "scene_type": "warehouse",
    "stats": {
        "person_count": 3,
        "vehicle_count": 2,
        "obstacle_count": 0
    },
    "objects": [
        {
            "track_id": 1,
            "class_name": "person",
            "confidence": 0.95,
            "bbox": [x1, y1, x2, y2]
        }
    ]
}
```

### SSE 实时推送

```bash
# 实时感知流
GET /sse/perception
```

## 项目结构

```
.
├── logistics_perception.py    # 核心感知引擎
├── logistics_server.py        # FastAPI 服务
├── templates/
│   └── logistics_demo.html    # Web 演示界面
├── requirements_logistics.txt  # 依赖
└── test_logistics.py          # 测试脚本
```

## 使用示例

### Python API

```python
from logistics_perception import LogisticsPerception

# 初始化
perception = LogisticsPerception()
perception.load_models()

# 处理帧
result = perception.understand_scene(frame)

# 获取结果
print(f"场景: {result.scene_type}")
print(f"检测到 {len(result.objects)} 个物体")
for obj in result.objects:
    print(f"  - {obj.class_name} (ID:{obj.track_id})")
```

### 命令行测试

```bash
# 测试检测
python -c "from logistics_perception import LogisticsPerception; \
p = LogisticsPerception(); \
p.load_models(); \
import cv2; \
cap = cv2.VideoCapture(0); \
ret, frame = cap.read(); \
result = p.understand_scene(frame); \
print(f'Detected: {len(result.objects)} objects')"
```

## 支持的类别

### 仓库场景
| 类别 | 说明 |
|-----|------|
| person | 人员 |
| forklift | 叉车 |
| pallet | 托盘 |
| box | 货物箱 |
| shelf | 货架 |
| conveyor | 传送带 |
| robot | AGV/AMR机器人 |

### 无人车场景
| 类别 | 说明 |
|-----|------|
| vehicle | 车辆 |
| obstacle | 障碍物 |
| traffic_sign | 交通标识 |
| lane | 车道线 |
| danger_zone | 危险区域 |

## 性能优化建议

### GPU 加速

```python
# 检测是否支持 CUDA
import torch
if torch.cuda.is_available():
    config["detector"]["device"] = "cuda"
```

### 模型选择

```python
# 轻量级 (快但精度较低)
model = "yolov8n.pt"

# 平衡
model = "yolov8s.pt"

# 高精度
model = "yolov8m.pt"
```

### 跳帧处理

```python
# 高分辨率视频，降低处理频率
config["skip_frames"] = 3  # 每3帧处理1次
```

## 扩展功能

### 集成 SAM 分割

```python
from segment_anything import sam_model_registry

class LogisticsPerception:
    def _load_segmenter(self):
        sam = sam_model_registry["vit_h"]("sam_vit_h.pth")
        sam.to(device="cuda")
        self._sam = sam
```

### 集成 DeepSort 追踪

```python
from deep_sort_realtime.deepsort_tracker import DeepSort

class LogisticsPerception:
    def __init__(self):
        self._deepsort = DeepSort(max_age=30)
```

## License

MIT License
