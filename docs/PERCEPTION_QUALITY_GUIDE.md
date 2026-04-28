# 三维感知系统 - 可靠性保障指南

## 📋 如何保证感知算法能用好用

### 1. 感知质量保障体系

```
┌─────────────────────────────────────────────────────────────┐
│                    感知可靠性 = 三层保障                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  第一层: 输入保障 (数据层)                                    │
│  ├── 传感器状态监控 (温度、曝光、帧率)                         │
│  ├── 数据完整性检查 (丢帧、损坏、异常值)                       │
│  └── 时序一致性 (时间戳同步、延迟监控)                         │
│                                                             │
│  第二层: 算法保障 (模型层)                                     │
│  ├── 模型质量评估 (精度、召回率)                               │
│  ├── 置信度校准 (概率输出是否可靠)                             │
│  └── 异常检测 (分布外样本识别)                                │
│                                                             │
│  第三层: 输出保障 (结果层)                                     │
│  ├── 时序滤波 (平滑、异常值剔除)                              │
│  ├── 物理约束检查 (尺寸、速度、加速度)                         │
│  └── 多传感器一致性检验                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. 核心质量指标

| 指标 | 定义 | 阈值 | 处理措施 |
|------|------|------|----------|
| **FPS** | 帧率 | ≥15fps | 降级到轻量模型 |
| **Latency** | 延迟 | ≤200ms | 跳过非关键计算 |
| **Valid Ratio** | 有效深度比 | ≥30% | 切换深度估计方法 |
| **Tracking Lost** | 追踪丢失率 | ≤10% | 重新初始化追踪 |
| **Detection Rate** | 检测率 | ≥80% | 降低检测阈值 |

### 3. 感知自检机制

```python
class PerceptionHealth:
    """感知系统健康状态"""
    
    def is_healthy(self) -> bool:
        """综合判断是否健康"""
        return (
            self.overall != QualityLevel.FAILED and  # 未完全失败
            len(self.errors) == 0 and               # 无错误
            self.camera_ok and                       # 相机正常
            self.depth_ok                            # 深度正常
        )
    
    def get_fallback_strategy(self) -> str:
        """获取降级策略"""
        if self.overall == QualityLevel.FAILED:
            return "FALLBACK_TO_SAFE_MODE"  # 进入安全模式
        elif self.overall == QualityLevel.POOR:
            return "REDUCE_COMPLEXITY"       # 降低复杂度
        elif self.overall == QualityLevel.ACCEPTABLE:
            return "INCREASE_FILTERING"      # 增加滤波
        else:
            return "FULL_CAPABILITY"          # 全功能运行
```

---

## 🔧 底层逻辑依赖

### 1. 核心依赖树

```
感知系统
│
├── 🎯 目标检测 (2D)
│   ├── 基础: numpy, opencv-python
│   ├── 模型: ultralytics (YOLOv8)
│   ├── 推理: torch, torchvision
│   └── 追踪: supervision (ByteTrack)
│
├── 📐 深度估计 (3D)
│   ├── 方法1: 单目深度 (MiDaS/DepthPro)
│   │   └── torch >= 2.0
│   ├── 方法2: 双目匹配 (SGBM/RAFT-Stereo)
│   │   └── opencv-contrib
│   └── 方法3: RGB-D融合
│       └── pyorbbecsdk / librealsense2
│
├── 🗺️ 点云处理
│   ├── 生成: reprojectImageTo3D
│   ├── 处理: open3d
│   └── 可视化: open3d / matplotlib
│
├── 📊 质量保障
│   ├── 时序滤波: scipy.signal
│   ├── 异常检测: sklearn
│   └── 性能监控: psutil
│
└── 🤖 传感器融合
    ├── IMU融合: imu-tools / pyquaternion
    └── GPS融合: numpy
```

### 2. 安装依赖

```bash
# 核心依赖
pip install numpy opencv-python

# 目标检测
pip install ultralytics torch torchvision

# 点云处理
pip install open3d

# 追踪
pip install supervision

# 深度估计 (可选)
pip install torch timm

# 性能监控
pip install psutil

# ROS集成 (可选)
pip install rospkg sensor_msgs
```

### 3. 硬件依赖

| 传感器 | 型号 | 用途 | 优先级 |
|--------|------|------|--------|
| **RGB相机** | 任意 | 2D检测 | 必须 |
| **深度相机** | RealSense D455 | RGB-D感知 | 推荐 |
| **双目相机** | ZED 2i | 双目深度 | 推荐 |
| **激光雷达** | Ouster/Livox | 精确3D感知 | 高级 |
| **IMU** | BMI088 | 运动融合 | 可选 |

---

## 🏗️ 三维感知架构

### 1. 系统架构

```
输入层
  │
  ├── RGB图像 ──────▶ 2D目标检测 ──┐
  │                              │
  │                              ▼
  │                      ┌──────────────┐
  │                      │   传感器融合  │
  │                      └──────────────┘
  │                              │
  ├── 深度图 ────────────▶ 深度估计 ──┘
  │                              │
  │                              ▼
  │                      ┌──────────────┐
  │                      │  点云生成     │
  │                      └──────────────┘
  │                              │
  ├── 双目图像 ──────────▶ 双目匹配 ────┘
  │                              │
  │                              ▼
  │                      ┌──────────────┐
  └──────────────────────▶│  3D边界框估计 │
                          └──────────────┘
                                   │
                                   ▼
                          ┌──────────────┐
                          │    追踪      │
                          └──────────────┘
                                   │
                                   ▼
                          ┌──────────────┐
                          │  质量保障    │
                          └──────────────┘
                                   │
                                   ▼
                               输出结果
```

### 2. 深度估计方法对比

| 方法 | 精度 | 速度 | 硬件要求 | 适用场景 |
|------|------|------|----------|----------|
| **MiDaS (单目)** | ★★★ | ★★ | GPU | 通用 |
| **DepthPro (单目)** | ★★★★ | ★★ | GPU | 高精度 |
| **SGBM (双目)** | ★★★ | ★★ | 双目相机 | 室内 |
| **RAFT-Stereo** | ★★★★ | ★ | GPU+双目 | 室外 |
| **RGB-D融合** | ★★★★★ | ★★★★★ | 深度相机 | 室内机器人 |

### 3. 3D感知质量检查清单

```python
def quality_checklist(detections: List[DetectedObject3D], 
                      health: PerceptionHealth) -> Dict:
    """质量检查清单"""
    
    results = {
        # ✅ 必须通过
        "input_ok": health.camera_ok and health.depth_ok,
        "output_valid": len(detections) >= 0,
        
        # ⚠️ 建议检查
        "reasonable_count": len(detections) < 100,  # 异常多检测
        "reasonable_depth": all(0 < d.depth < 100 for d in detections),
        "reasonable_size": all(all(0 < s < 10 for s in d.size_3d) 
                              for d in detections if d.size_3d),
        
        # 📊 性能指标
        "fps_sufficient": health.fps >= 15,
        "latency_acceptable": health.latency_ms <= 200,
    }
    
    # 评分
    passed = sum(results.values())
    total = len(results)
    score = passed / total * 100
    
    return {
        "passed": passed,
        "total": total,
        "score": f"{score:.0f}%",
        "details": results,
        "is_reliable": results["input_ok"] and score >= 60
    }
```

---

## 🚨 常见问题与解决

### 1. 深度估计不准确

**原因**: 单目深度模型泛化差、光照变化大

**解决**:
```python
# 策略1: 切换到双目
perception = Perception3D(camera_type="stereo")

# 策略2: 使用RGB-D
depth_data = realsense.capture_depth()
detections = perception.update(rgb, depth=depth_data)

# 策略3: 增加时序滤波
depth_estimator = DepthEstimator(method="monocular")
depth_estimator._temporal_filter(depth, confidence)  # 使用历史平滑
```

### 2. 检测率低

**原因**: 目标遮挡、尺度变化、模型不匹配

**解决**:
```python
# 策略1: 降低阈值
config["detector"]["conf_threshold"] = 0.25

# 策略2: 使用数据增强的模型
model = YOLO("yolov8s.pt")  # 使用更大模型

# 策略3: 添加后处理
detections = apply_nms(detections, iou_threshold=0.5)
```

### 3. 延迟过高

**原因**: 模型太大、帧率过高、CPU瓶颈

**解决**:
```python
# 策略1: 使用轻量模型
model = YOLO("yolov8n.pt")  # nano版本

# 策略2: 跳帧处理
if frame_count % 2 == 0:  # 每2帧检测一次
    detections = detect(frame)

# 策略3: 异步处理
Thread(target=background_inference, args=(frame,)).start()
```

---

## 📈 推荐的感知配置

### 入门配置 (仅RGB相机)
```python
perception = Perception3D(camera_type="monocular")
# 依赖: numpy, opencv, ultralytics, torch
# 精度: ★★★
# 速度: ★★★
```

### 标准配置 (RGB + 深度相机)
```python
perception = Perception3D(camera_type="rgbd")
# 依赖: + librealsense2 / pyorbbecsdk
# 精度: ★★★★
# 速度: ★★★★★
```

### 高精度配置 (双目相机)
```python
perception = Perception3D(camera_type="stereo")
# 依赖: + opencv-contrib, ZED SDK
# 精度: ★★★★
# 速度: ★★★
```

### 最高精度配置 (多传感器融合)
```python
# 使用unified_perception_slam.py的FusionSLAM
system = create_unified_system("fusion")
# 融合: 视觉SLAM + 激光SLAM + 深度估计
# 精度: ★★★★★
# 速度: ★★
```

---

## ✅ 验证清单

在部署前，确保:

- [ ] **数据完整性**: 输入图像无损坏、丢帧
- [ ] **模型加载**: 检测器、深度模型正常加载
- [ ] **FPS达标**: ≥15fps (实时应用) 或 ≥5fps (离线处理)
- [ ] **延迟可控**: ≤200ms
- [ ] **深度有效**: Valid Ratio ≥30%
- [ ] **检测合理**: 无异常多/少检测
- [ **追踪稳定**: 目标ID连续性良好
- [ ] **物理约束**: 尺寸、速度在合理范围
- [ ] **降级策略**: 低质量时有备用方案
