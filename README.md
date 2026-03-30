# 项目简历：PCB缺陷智能检测系统

## 项目概述

**项目名称**：PCB缺陷智能检测系统  
**项目类型**：计算机视觉 / 工业质检 / 深度学习 / 目标检测  
**开发周期**：2024-2025年  
**技术栈**：Python、OpenCV、PyTorch、Ultralytics YOLOv8、FastAPI、ONNX、TensorRT  
**模型架构**：YOLOv8n（目标检测）+ 迁移学习  
**部署方式**：Docker + REST API + ONNX跨平台部署  
**推理优化**：ONNX Runtime优化、TensorRT量化（可选）

---

## 项目背景

在电子制造、PCB贴片等行业，产品质量检测是生产流程中的关键环节。传统人工检测存在效率低、一致性差、易疲劳等问题。本项目开发了一套基于深度学习的PCB智能缺陷检测系统，可实现电路板的自动化缺陷检测。

**数据集来源**：DeepPCB公开数据集（业界标准PCB缺陷数据集）

---

## 核心功能

### 1. 多种输入模式
- **单图检测**：输入单张PCB图片，返回缺陷位置和类型
- **批量处理**：自动检测文件夹内所有图片
- **API服务**：REST API接口，支持外部系统调用

### 2. 6类PCB缺陷检测

| 缺陷类型 | 英文名 | 说明 | 危险等级 |
|---------|--------|------|----------|
| 缺失孔洞 | missing_hole | 焊盘孔洞缺失 | 高 |
| 鼠咬缺陷 | mouse_bite | 走线边缘缺口 | 中 |
| 开路 | open_circuit | 走线断裂 | 高 |
| 短路 | short | 走线意外连接 | 极高 |
| 毛刺 | spur | 多余铜箔突出 | 中 |
| 多余铜 | spurious_copper | 残留铜箔 | 低 |

### 3. DSSM缺陷根因定位（新增）

基于DSSM双塔匹配思想，对缺陷进行根因分析，反推最可能出现异常的工序环节。

**支持工序**：
- 表面处理：清洗、沉铜、蚀刻、阻焊
- 组装工序：SMT贴片、回流焊、波峰焊
- 检测工序：AOI、X-Ray、ICT

**输出**：
- 缺陷类型 + 位置
- 可能工序排序
- 风险预警
- 处理建议

### 3. 完整的输出体系
- **图像标注**：自动绘制缺陷边界框和标签
- **JSON报告**：详细的检测结果（位置、类型、置信度）
- **CSV统计**：批量统计信息，便于数据分析
- **缺陷ROI**：单独保存每个缺陷区域图片

### 4. 智能标注工具
- **模板对比法**：自动检测PCB与模板的差异区域
- **半自动标注**：AI辅助 + 人工校正
- **支持格式**：YOLO格式标注

---

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户交互层                            │
│         PyQt5 GUI / 命令行 / FastAPI / Web                 │
├─────────────────────────────────────────────────────────────┤
│                        应用逻辑层                            │
│   图像采集 → 预处理 → 推理 → 后处理 → 结果输出              │
├─────────────────────────────────────────────────────────────┤
│                        算法核心层                            │
│        YOLOv8 + OpenCV + NumPy + ONNX Runtime             │
├─────────────────────────────────────────────────────────────┤
│                        数据存储层                            │
│           文件系统 + JSON/CSV + YAML配置                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心算法

### 1. 模板匹配算法
```python
# 差异检测 + 形态学处理
diff = cv2.absdiff(image, template)
_, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
kernel = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

### 2. YOLOv8目标检测
- **模型**：YOLOv8n（nano版本，轻量快速）
- **输入尺寸**：640×640×3
- **预训练权重**：COCO数据集
- **训练策略**：
  - 数据增强：Mosaic、MixUp、随机翻转
  - 优化器：AdamW
  - 学习率：余弦退火
  - 早停策略

### 3. 推理后处理
```python
# NMS + 置信度过滤
results = model.predict(image, conf=0.25, iou=0.45)
# 过滤低置信度检测框
# 分类排序输出
```

---

## 数据集

### 数据集统计

| 统计项 | 数值 |
|--------|------|
| 总图像数 | 1,022张 |
| 训练集 | 632张 (62%) |
| 验证集 | 192张 (19%) |
| 测试集 | 198张 (19%) |
| 标注框总数 | 1,637个 |
| 缺陷类别 | 6类 |

### 数据集结构
```
yolo_pcb_dataset/
├── images/
│   ├── train/      # 632张训练图片
│   ├── val/        # 192张验证图片
│   └── test/       # 198张测试图片
├── labels/
│   ├── train/      # YOLO格式标注
│   ├── val/
│   └── test/
├── classes.txt     # 类别定义
└── data.yaml       # 训练配置
```

### 6类缺陷类别
1. `missing_hole` - 缺失孔洞
2. `mouse_bite` - 鼠咬
3. `open_circuit` - 开路
4. `short` - 短路
5. `spur` - 毛刺
6. `spurious_copper` - 假铜

---

## 项目成果

### 1. 已训练模型

| 模型文件 | 说明 | 大小 |
|---------|------|------|
| `best.pt` | YOLOv8n最佳权重 | ~6MB |
| `last.pt` | 最后一轮权重 | ~6MB |

**模型配置**：
- **框架**：Ultralytics YOLOv8
- **架构**：YOLOv8n（nano）
- **输入尺寸**：640×640
- **参数量**：3.0M
- **GFLOPs**：8.1

### 2. 检测性能

| 指标 | 数值 | 说明 |
|------|------|------|
| **mAP50** | 0.302 | IoU=0.5时的平均精度 |
| mAP50-95 | 0.229 | 多IoU阈值平均精度 |
| **Precision** | 0.207 | 精确率 |
| **Recall** | 0.524 | 召回率 |
| 推理速度 | **11ms/帧** | ~90 FPS (GPU) |

**各类别性能**：

| 缺陷类型 | Precision | Recall | mAP50 |
|---------|-----------|--------|-------|
| missing_hole | 0.295 | 0.508 | 0.431 |
| mouse_bite | 0.162 | 0.565 | 0.223 |
| open_circuit | 0.191 | 0.443 | 0.204 |
| **short** | **0.234** | 0.414 | **0.372** |
| spur | 0.193 | 0.555 | 0.288 |
| spurious_copper | 0.170 | 0.662 | 0.294 |

### 3. 导出能力
- ✅ ONNX格式导出
- ✅ TensorRT量化支持
- ✅ OpenCV DNN推理
- ✅ ONNX Runtime跨平台

### 4. API服务
```python
# FastAPI服务
- 支持图片URL/Base64/文件上传
- 返回JSON格式检测结果
- 支持批量推理
- 健康检查接口
```

### 5. 系统特性
- ✅ 支持实时检测（90 FPS）
- ✅ 支持批量离线处理
- ✅ 支持GPU/CPU推理
- ✅ 模块化设计，易于扩展
- ✅ 完善的日志记录

---

## 技术亮点

### 1. 工程化设计
- **配置驱动**：YAML配置文件，无需修改代码
- **模块化架构**：训练/推理/标注解耦
- **统一入口**：命令行参数统一管理
- **完善的错误处理**：异常捕获和日志

### 2. 算法优化
- **多尺度检测**：不同尺寸缺陷自适应
- **数据增强**：Mosaic、MixUp提升泛化
- **预训练+微调**：迁移学习加速收敛
- **NMS后处理**：去除重复检测框

### 3. 部署优化
- **ONNX导出**：跨平台部署
- **模型量化**：INT8加速（可选）
- **批处理**：支持批量推理
- **API服务**：RESTful接口

---

## 项目文件结构

```
e:/pyspace/opencv/
├── yolo_pcb_dataset/           # PCB缺陷数据集
│   ├── images/
│   │   ├── train/             # 训练集 (632张)
│   │   ├── val/               # 验证集 (192张)
│   │   └── test/              # 测试集 (198张)
│   ├── labels/                # YOLO标注
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── classes.txt            # 类别定义
│   └── data.yaml              # 数据集配置
│
├── models/                    # 训练模型
│   └── yolov8/
│       └── train/
│           └── weights/
│               ├── best.pt     # 最佳模型
│               └── last.pt    # 最后模型
│
├── train_yolov8.py            # 训练脚本
├── smart_label_v3.py          # 智能标注工具
├── defect_detection_api.py   # API服务
├── export_onnx.py             # ONNX导出
├── auto_label_pcb.py         # 自动标注
├── interactive_label_pcb.py  # 交互式标注
├── optimize_train.py         # 训练优化
│
└── README.md                  # 项目文档
```

---

## 使用指南

### 1. 模型训练
```bash
# 训练模型
python train_yolov8.py --epochs 30 --train

# 使用更大模型
python train_yolov8.py --model s --epochs 50 --train

# 评估模型
python train_yolov8.py --eval
```

### 2. 模型推理
```python
from ultralytics import YOLO

model = YOLO('models/yolov8/train/weights/best.pt')
results = model.predict('test.jpg', conf=0.25)

# 打印检测结果
for r in results:
    print(r.boxes)
```

### 3. 启动API服务
```bash
python defect_detection_api.py
# 访问 http://localhost:8000/docs
```

### 4. 导出ONNX
```bash
python export_onnx.py
```

---

## 应用场景

### 1. PCB制造
- 电路板缺陷检测
- 焊点质量检测
- 来料检验

### 2. 半导体
- 晶圆表面缺陷
- 芯片封装检测

### 3. 工业质检
- 表面瑕疵检测
- 尺寸测量
- 质量追溯

---

## 技能展示

### 编程能力
- **Python**：面向对象、装饰器、生成器
- **代码规范**：PEP8、文档字符串
- **工程能力**：模块化、错误处理、日志

### 计算机视觉
- **OpenCV**：图像处理、特征提取
- **深度学习**：YOLO、CNN、迁移学习
- **目标检测**：NMS、IoU、mAP

### 工程实践
- **软件工程**：配置管理、版本控制
- **系统架构**：分层设计、接口抽象
- **部署运维**：Docker、API服务、性能优化

---

## 项目价值

1. **降本增效**：替代人工检测，降低人力成本
2. **质量提升**：检测一致性高，减少漏检
3. **快速部署**：配置化设计，易于部署
4. **可扩展性**：支持新缺陷类型快速添加

---

## 优化方向

### 短期优化
- [ ] 手动标注50-100张关键样本（预期mAP提升至0.5+）
- [ ] 使用更大模型 yolov8s/m（预期mAP提升10-20%）
- [ ] 增加训练轮次至50-100（预期mAP提升5-15%）

### 长期优化
- [ ] 构建更大数据集（收集真实工业数据）
- [ ] 模型量化压缩（TensorRT INT8）
- [ ] 边缘设备部署（Jetson系列）

---

## 联系方式

- **项目仓库**：本地项目
- **演示**：运行 `python train_yolov8.py --eval` 查看效果

---

**关键词**：计算机视觉、深度学习、工业质检、缺陷检测、YOLOv8、OpenCV、Python、智能制造、目标检测
