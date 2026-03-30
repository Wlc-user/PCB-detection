# 🎯 PCB缺陷检测系统 - 作品集

## 项目概述

基于YOLOv8的PCB表面缺陷实时检测系统，结合DSSM双塔匹配算法实现缺陷根因定位，反推异常工序环节。

---

## 一、技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        系统架构图                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│   │   数据采集   │───▶│  YOLOv8检测  │───▶│ DSSM根因分析 │    │
│   └──────────────┘    └──────────────┘    └──────────────┘    │
│         │                   │                   │              │
│         ▼                   ▼                   ▼              │
│   相机/激光雷达       6类缺陷检测         工序定位           │
│                                                                 │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│   │  图像预处理  │    │   API服务    │    │   部署上线   │    │
│   └──────────────┘    └──────────────┘    └──────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、工控级特性

| 功能 | 说明 | 状态 |
|------|------|------|
| **PLC通信** | Modbus TCP对接产线 | ✅ 已实现 |
| **模型量化** | INT8加速2-4倍 | ✅ 已实现 |
| **ONNX导出** | 跨平台部署 | ✅ 已实现 |
| **看门狗** | 故障自动恢复 | ⏳ 待部署 |
| **日志记录** | 问题追溯 | ✅ 已实现 |

### 2.1 PLC通信 (Modbus TCP)

```
检测结果 → PLC寄存器
- 4000: 检测状态 (0=空闲, 1=检测中, 2=完成)
- 4001: 缺陷数量
- 4002~4017: 缺陷类型+置信度
```

### 2.2 模型优化

```bash
# 量化加速
python optimize_train.py

# 导出ONNX
python export_onnx.py
```

---

## 三、核心功能

### 2.1 缺陷检测

| 缺陷类型 | 英文名 | 危险等级 |
|---------|--------|----------|
| 缺失孔洞 | missing_hole | 🔴 高 |
| 鼠咬缺陷 | mouse_bite | 🟡 中 |
| 开路 | open_circuit | 🔴 高 |
| 短路 | short | ⚫ 极高 |
| 毛刺 | spur | 🟡 中 |
| 多余铜 | spurious_copper | 🟢 低 |

### 2.2 根因定位（DSSM双塔匹配）

```
缺陷特征 ──┐
           ├─▶ 双塔匹配 ──▶ 工序根因
工序特征 ──┘
```

**支持工序**：
- 表面处理：清洗、沉铜、蚀刻、阻焊
- 组装：SMT贴片、回流焊、波峰焊
- 检测：AOI、X-Ray、ICT

---

## 三、效果展示

### 3.1 缺陷检测示例

| 图片 | 说明 |
|------|------|
| `portfolio/result_1.jpg` | 测试图片1 - 缺失孔洞检测 |
| `portfolio/result_2.jpg` | 测试图片2 - 多余铜检测 |
| `portfolio/result_3.jpg` | 测试图片3 - 缺失孔洞检测 |

### 3.2 API服务

| 接口 | 地址 |
|------|------|
| 缺陷检测API | http://localhost:8000/docs |
| 根因分析API | http://localhost:8001/docs |

---

## 四、技术栈

| 层级 | 技术 |
|------|------|
| 深度学习 | PyTorch, YOLOv8 |
| 图像处理 | OpenCV, NumPy |
| API服务 | FastAPI, Uvicorn |
| 部署 | Docker, ONNX |
| 标定技术 | PnP, ICP, LM优化 |

---

## 五、项目文件

```
e:/pyspace/opencv/
├── 核心代码
│   ├── train_yolov8.py          # 模型训练
│   ├── defect_detection_api.py  # 缺陷检测API
│   ├── defect_root_cause.py     # DSSM根因分析
│   ├── root_cause_api.py        # 根因API
│   └── inference_demo.py        # 推理演示
│
├── 配置文件
│   ├── configs/train_config.yaml
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── 数据集
│   └── yolo_pcb_dataset/        # 1022张PCB图片
│
├── 作品集
│   └── portfolio/               # 效果展示图片
│
└── 文档
    ├── README.md                # 项目文档
    └── DEPLOY.md                # 部署文档
```

---

## 六、运行指南

### 6.1 启动服务

```bash
# 缺陷检测API
python defect_detection_api.py

# 根因分析API
python root_cause_api.py

# 交互式演示
python run_demo.py
```

### 6.2 API调用示例

```python
import requests

# 缺陷检测
with open('test.jpg', 'rb') as f:
    r = requests.post('http://localhost:8000/detect', files={'file': f})
    print(r.json())

# 根因分析
with open('test.jpg', 'rb') as f:
    r = requests.post('http://localhost:8001/analyze', files={'file': f})
    print(r.json())
```

---

## 七、项目亮点

- ✅ YOLOv8实时缺陷检测
- ✅ DSSM双塔匹配根因定位
- ✅ 支持6类PCB缺陷
- ✅ REST API服务化
- ✅ Docker部署支持
- ✅ 激光雷达标定技术
- ✅ 完整的项目文档

---

## 八、优化方向

1. **手动标注50-100张** → mAP提升至0.5+
2. **模型量化** → 推理速度提升2-4倍
3. **边缘部署** → Jetson/手机端部署

---

**项目状态**: Demo/POC阶段，可直接运行演示
**联系方式**: 13350514620
