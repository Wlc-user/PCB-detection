# PCB缺陷检测 - 技术Demo使用指南

## 项目概述

这是一个基于深度学习的PCB缺陷检测系统，技术Demo用于展示：
- YOLOv8目标检测技术
- OpenCV图像处理
- FastAPI接口服务
- 模型部署能力

---

## 快速开始

### 1. 环境准备
```bash
pip install -r requirements.txt
```

### 2. 运行Demo
```bash
# 方式1：交互式Demo菜单
python run_demo.py

# 方式2：直接推理
python inference_demo.py --image "yolo_pcb_dataset/images/test/01_missing_hole_02.jpg"
```

---

## Demo演示内容

| 演示 | 命令 | 说明 |
|------|------|------|
| 单图检测 | `python inference_demo.py --image <图片>` | 检测单张PCB图片 |
| 批量检测 | `python inference_demo.py --dir <文件夹>` | 批量检测 |
| API服务 | `python defect_detection_api.py` | 启动REST API |
| 模型评估 | `python train_yolov8.py --eval` | 查看性能指标 |
| ONNX导出 | `python export_onnx.py` | 导出模型 |

---

## 效果展示

### 检测结果示例

输入图片 → 输出带标注的图片 + JSON结果

```json
{
  "image": "01_missing_hole_02.jpg",
  "detections": [
    {
      "class": "missing_hole",
      "confidence": 0.78,
      "bbox": [x1, y1, x2, y2]
    }
  ]
}
```

---

## 技术亮点

- ✅ YOLOv8 实时目标检测
- ✅ 支持6类PCB缺陷检测
- ✅ REST API 接口服务
- ✅ ONNX 跨平台部署
- ✅ 模板对比自动标注

---

## 项目结构

```
.
├── train_yolov8.py          # 训练脚本
├── inference_demo.py         # 推理演示
├── defect_detection_api.py  # API服务
├── export_onnx.py           # ONNX导出
├── smart_label_v3.py        # 智能标注
├── run_demo.py              # Demo菜单
├── README.md                # 项目文档
└── requirements.txt         # 依赖
```

---

## 注意事项

> ⚠️ 当前模型使用DeepPCB公开数据集训练
> 
> 实际应用需要：
> 1. 收集真实PCB数据
> 2. 手动标注缺陷
> 3. 重新训练模型

---

## 联系方式

- 项目类型：技术Demo / POC
- 技术栈：Python + OpenCV + PyTorch + FastAPI
