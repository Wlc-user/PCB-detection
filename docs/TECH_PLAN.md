# PCB 缺陷检测技术方案

> 生成时间: 2026-04-08  
> 目标: 建立完整的视觉任务解决方案，覆盖检测/分割/关键点/追踪等场景

---

## 一、任务类型与架构选型

### 1.1 任务类型总览

| 任务类型 | 典型场景 | 推荐架构 | 复杂度 |
|---------|---------|---------|-------|
| **目标检测** | 缺陷定位、计数 | YOLOv8/v10、RT-DETR | ⭐⭐ |
| **实例分割** | 精确边界、面积测量 | YOLOv8-Seg、Mask R-CNN | ⭐⭐⭐ |
| **语义分割** | 整图区域划分 | U-Net、DeepLabV3+ | ⭐⭐⭐ |
| **关键点检测** | Pin 脚定位、焊点检测 | HRNet、YOLO-Pose | ⭐⭐⭐ |
| **目标追踪** | 流水线多帧跟踪 | ByteTrack、OC-SORT | ⭐⭐⭐⭐ |
| **图像超分** | 低分辨率 PCB 增强 | ESRGAN、Real-ESRGAN | ⭐⭐⭐ |
| **图像分类** | 良品/次品粗筛 | EfficientNet、ViT | ⭐ |

---

### 1.2 PCB 缺陷检测模型选型

```
                    ┌─────────────────────────────────────┐
                    │           任务复杂度                │
                    └─────────────────────────────────────┘
                                ↑
          实时性要求高 ──────────────────── 精度要求高
                                ↓
        ┌───────────┐    ┌───────────┐    ┌───────────┐
        │  YOLOv8n  │    │  YOLOv8s  │    │ RT-DETR   │
        │  (nano)   │    │  (small)  │    │  (高精度)  │
        └───────────┘    └───────────┘    └───────────┘
              ↑              ↑               ↑
         边缘设备       工业相机         高端检测台
         (Jetson)       (GPU服务器)      (A100)
```

#### 选型决策树

```
输入图像 → 需要实例级边界?
    │
    ├── 否 → 需要实时跟踪?
    │       ├── 否 → YOLOv8 检测 (当前方案 ✓)
    │       └── 是 → YOLOv8 + ByteTrack
    │
    └── 是 → 需要像素级精度?
            ├── 否 → YOLOv8-Seg 分割头
            └── 是 → Mask R-CNN / SAM + 后处理
```

#### 模型对比

| 模型 | mAP@50 | 推理速度 (FP32) | 模型大小 | 适用场景 |
|------|--------|-----------------|---------|---------|
| YOLOv8n | 37.3 | **2.5ms** | 6.3MB | 边缘部署 |
| YOLOv8s | 44.9 | 4.0ms | 22MB | 工业相机 |
| YOLOv8m | 50.2 | 6.5ms | 52MB | 服务器 |
| RT-DETR-H | 56.1 | 12ms | 435MB | 高精度场景 |
| YOLOv10-S | 46.3 | 3.2ms | 24MB | 新方案 |

---

## 二、数据增强策略

### 2.1 增强层级

```
┌────────────────────────────────────────────────────────────┐
│                     数据增强 Pipeline                       │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   │
│  │  几何    │ → │  光学    │ → │  高级   │ → │  Mixup  │   │
│  │  变换    │   │  变换    │   │  增强   │   │  系列   │   │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘   │
│                                                             │
│  • 旋转 (±15°)        • 亮度 ±30%         • CutOut        • Mosaic 1.0    │
│  • 翻转 (水平)        • 对比度 ±20%        • CopyPaste    • MixUp 0.15   │
│  • 缩放 (0.5-1.5)     • 饱和度 ±30%        • GridMask     • CopyPaste    │
│  • 平移 (±10%)        • 添加噪声           • RandomErasing 0.1         │
│  • 仿射变换           • 模糊 (运动/高斯)                    • CutMix 0.1   │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### 2.2 PCB 场景专用增强

```python
# pcb_augmentation.py - PCB专用增强策略

class PCBSpecificAugmentation:
    """
    针对 PCB 缺陷的专用增强
    """
    
    # 1. PCB 特有的几何畸变
    def pcb_perspective_distortion(self, img, mask):
        """PCB 轻微透视畸变 (拍摄角度导致)"""
        # 模拟 5-10 度倾斜
        angle = random.uniform(-10, 10)
        # ...
    
    # 2. 光照不均匀增强
    def uneven_illumination(self, img):
        """模拟 PCB 检测中的光照不均"""
        # 创建渐变遮罩
        gradient = self.create_radial_gradient(img.shape)
        # 局部亮度调节
    
    # 3. 电路纹理干扰
    def circuit_texture_interference(self, img, bbox):
        """添加背景电路纹理作为干扰"""
        # 从背景区域提取纹理
        texture = self.extract_circuit_pattern(img)
        # 混合到目标区域
    
    # 4. 缺陷形态学变换
    def defect_morphology(self, mask, defect_type):
        """针对不同缺陷类型的形态学增强"""
        if defect_type == 'missing_hole':
            # 孔洞边缘毛刺
            return morphology.dilation(mask, disk(1))
        elif defect_type == 'mouse_bite':
            # 啃噬边缘不规则化
            return morphology.irregular_boundary(mask)
```

### 2.3 增强强度推荐

| 缺陷类型 | Mosaic | MixUp | 旋转范围 | 亮度范围 |
|---------|--------|-------|---------|---------|
| missing_hole | 1.0 | 0.15 | ±5° | ±20% |
| mouse_bite | 0.8 | 0.1 | ±15° | ±30% |
| open_circuit | 1.0 | 0.2 | ±5° | ±25% |
| short | 0.5 | 0.1 | ±10° | ±15% |
| spur | 0.8 | 0.15 | ±20° | ±30% |
| spurious_copper | 0.8 | 0.1 | ±10° | ±25% |

---

## 三、Loss 设计方案

### 3.1 检测任务 Loss

```
                    ┌─────────────────────────────────┐
                    │         Total Loss              │
                    │  L_total = L_box + L_cls + L_dfl │
                    └─────────────────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          ↓                             ↓                             ↓
    ┌───────────┐               ┌───────────┐               ┌───────────┐
    │   Box     │               │  Class    │               │   DFL     │
    │  Loss     │               │  Loss     │               │  Loss     │
    │ (CIoU)    │               │ (BCE)     │               │ (Distribution) │
    └───────────┘               └───────────┘               └───────────┘
          │                             │                             │
          ↓                             ↓                             ↓
    • 中心点距离                 • 前景/背景分类             • 边界框回归
    • 宽高比差异                 • 类别交叉熵                • 分布焦点损失
    • 重叠面积                   • 类别不平衡处理            • 4个边位置
```

### 3.2 分割任务 Loss

```python
# Segmentation Loss 组合
segmentation_loss = {
    'dice_loss': 1.0,      # 前景/背景重叠度
    'bce_loss': 1.0,        # 像素级交叉熵
    'iou_loss': 0.5,        # IoU损失
    'boundary_loss': 0.2,   # 边界损失 (精细边缘)
}

# 边界损失实现
class BoundaryLoss(nn.Module):
    """边界感知损失 - 提升边缘精度"""
    def forward(self, pred, target):
        # 计算梯度
        pred_grad = gradient(pred)
        target_grad = gradient(target)
        # 边界区域加权
        return F.mse_loss(pred_grad * boundary_mask, 
                          target_grad * boundary_mask)
```

### 3.3 关键点检测 Loss

```python
# 关键点检测 - OKS-based Loss
class OKSLoss(nn.Module):
    """
    基于 OKS (Object Keypoint Similarity) 的损失
    适用于 Pin 脚定位、焊点检测
    """
    def forward(self, pred_kpts, target_kpts, target_visibility):
        # 计算每个关键点的 OKS
        oks = self.compute_oks(pred_kpts, target_kpts)
        # 不可见关键点不参与损失
        visible_oks = oks * target_visibility
        return 1 - visible_oks.mean()
```

---

## 四、训练策略

### 4.1 分阶段训练

```
┌──────────────────────────────────────────────────────────────────┐
│                      训练 Pipeline                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Phase 1: 预训练 (Freeze backbone)        ~20 epochs             │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  Backbone: 冻结                                            │   │
│  │  Head: 可训练                                              │   │
│  │  lr: 1e-3 → 1e-4 (cosine decay)                           │   │
│  │  batch: 32                                                │   │
│  └────────────────────────────────────────────────────────────┘   │
│                              ↓                                    │
│  Phase 2: 全量微调 (Fine-tune all)        ~50 epochs             │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  All layers: 可训练                                        │   │
│  │  lr: 1e-4 → 1e-5 (cosine decay)                           │   │
│  │  batch: 16 (GPU memory limit)                             │   │
│  │  mosaic: 1.0 → 0.8 (gradually reduce)                     │   │
│  └────────────────────────────────────────────────────────────┘   │
│                              ↓                                    │
│  Phase 3: 蒸馏/量化感知训练 (可选)      ~20 epochs               │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  使用大模型作为教师                                        │   │
│  │  知识蒸馏损失: L_kd = KL(student_logits, teacher_logits)   │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 学习率调度

```python
# 推荐学习率策略

lr_schedule = {
    # Phase 1: 高学习率快速收敛
    'phase1': {
        'optimizer': 'AdamW',
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'warmup_epochs': 3,
        'scheduler': 'cosine',
    },
    
    # Phase 2: 降低学习率微调
    'phase2': {
        'optimizer': 'SGD',
        'lr': 1e-4,
        'weight_decay': 5e-4,
        'momentum': 0.937,
        'scheduler': 'cosine',
        'warmup_epochs': 2,
    },
    
    # Cosine Annealing with Warmup
    'cosine_warmup': {
        'T_max': 100,
        'eta_min': 1e-6,
        'warmup_epochs': 5,
    }
}
```

### 4.3 类别不平衡处理

```python
# PCB 缺陷类别分布可能不均衡

class ImbalancedDataSampler:
    """类别平衡采样器"""
    
    def __init__(self, dataset, mode='sqrt'):
        """
        Args:
            mode: 'sqrt' (平方根采样) / 'effective_num' (有效样本数)
        """
        self.class_counts = self.get_class_counts(dataset)
        self.class_weights = self.compute_weights(self.class_counts, mode)
    
    def compute_weights(self, counts, mode):
        if mode == 'sqrt':
            # 平方根采样: 减少高频类采样
            weights = np.sqrt(sum(counts) / counts)
        else:
            # Effective Number of Samples
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, counts)
            weights = (1 - beta) / effective_num
        
        return weights / weights.sum() * len(weights)
```

---

## 五、完整 Pipeline 架构

### 5.1 端到端训练流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PCB 缺陷检测完整 Pipeline                          │
└─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │   原始数据   │ → │   数据清洗   │ → │   数据标注   │ → │   数据划分   │
  │  (原始图片)  │     │  (去重/过滤) │     │  (LabelMe)  │     │  (8:1:1)   │
  └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                        │
        ┌───────────────────────────────────────────────────────────────┘
        ↓
  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │   数据加载   │ → │   数据增强   │ → │    模型      │ → │   损失计算   │
  │  (Dataloader)│    │  (Albumentations)│  │  (YOLOv8)  │     │  (CIoU+CE) │
  └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                        │
        ┌───────────────────────────────────────────────────────────────┘
        ↓
  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │  梯度反向   │ → │  优化器更新   │ → │  验证评估    │ → │   模型导出   │
  │  传播       │     │  (AdamW/SGD)│     │  (mAP/F1)   │     │ (ONNX/TRT) │
  └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                        │
        ┌───────────────────────────────────────────────────────────────┘
        ↓
  ┌─────────────────────────────────────────────────────────┐
  │                    部署推理阶段                          │
  │  ┌───────────┐   ┌───────────┐   ┌───────────┐         │
  │  │  图像预处理 │ → │   推理     │ → │  后处理    │         │
  │  │  (Resize) │   │ (ONNX/TRT)│   │ (NMS)     │         │
  │  └───────────┘   └───────────┘   └───────────┘         │
  └─────────────────────────────────────────────────────────┘
```

### 5.2 推理部署流程

```
                    推理请求
                       │
                       ↓
            ┌─────────────────────┐
            │   图像预处理         │
            │  • 缩放到 640x640    │
            │  • BGR → RGB        │
            │  • 归一化 [0,1]      │
            │  • HWC → NCHW       │
            └─────────────────────┘
                       │
                       ↓
            ┌─────────────────────┐
            │   模型推理           │
            │  ┌─────────────────┐│
            │  │ ONNX Runtime    ││
            │  │ TensorRT        ││
            │  │ OpenVINO        ││
            │  └─────────────────┘│
            └─────────────────────┘
                       │
                       ↓
            ┌─────────────────────┐
            │   后处理            │
            │  • 解码输出         │
            │  • NMS 去重         │
            │  • 坐标映射回原图    │
            └─────────────────────┘
                       │
                       ↓
            ┌─────────────────────┐
            │   结果输出          │
            │  • 缺陷类别         │
            │  • 置信度           │
            │  • 边界框坐标       │
            └─────────────────────┘
```

---

## 六、不同任务模块化设计

### 6.1 统一接口设计

```python
# task_router.py - 统一任务路由

class TaskRouter:
    """根据任务类型路由到对应模型"""
    
    TASK_MODELS = {
        'detection': {
            'model': 'YOLOv8',
            'backends': ['onnx', 'tensorrt', 'openvino'],
            'postprocess': 'nms',
        },
        'segmentation': {
            'model': 'YOLOv8-Seg',
            'backends': ['onnx', 'tensorrt'],
            'postprocess': 'mask_decode',
        },
        'pose': {
            'model': 'YOLO-Pose',
            'backends': ['onnx'],
            'postprocess': 'keypoint_decode',
        },
    }
    
    def __init__(self, task_type='detection', model_path=None):
        self.task_type = task_type
        self.config = self.TASK_MODELS[task_type]
        self.model = self.load_model(model_path)
    
    def predict(self, image):
        # 统一推理接口
        return self.model(image)
```

### 6.2 模块化组件

```
e:\pyspace\opencv\
├── tasks/
│   ├── __init__.py
│   ├── base.py              # 基类定义
│   ├── detector.py           # 检测任务
│   ├── segmenter.py          # 分割任务
│   ├── pose_estimator.py     # 关键点任务
│   └── super_resolution.py   # 超分任务
│
├── augmentations/
│   ├── __init__.py
│   ├── geometric.py          # 几何变换
│   ├── optical.py            # 光学变换
│   └── pcb_specific.py       # PCB专用
│
├── losses/
│   ├── __init__.py
│   ├── detection_loss.py     # 检测损失
│   ├── segmentation_loss.py  # 分割损失
│   └── focal_loss.py         # 焦点损失
│
├── optimizers/
│   ├── __init__.py
│   ├── lr_scheduler.py       # 学习率调度
│   └── warmup.py             # Warmup策略
│
├── data/
│   ├── __init__.py
│   ├── dataset.py            # 数据集类
│   ├── transforms.py          # 变换管道
│   └── samplers.py           # 采样器
│
└── deploy/
    ├── __init__.py
    ├── onnx_exporter.py       # ONNX导出
    ├── trt_builder.py         # TensorRT构建
    └── api_server.py          # API服务
```

---

## 七、推荐配置参数

### 7.1 PCB 检测最优配置

```yaml
# configs/pcb_detection_best.yaml

# 模型配置
model:
  name: yolov8s                    # 平衡精度与速度
  pretrained: true
  imgsz: 640
  conf: 0.25
  iou: 0.45

# 数据配置
data:
  train: yolo_pcb_dataset/images/train
  val: yolo_pcb_dataset/images/val
  test: yolo_pcb_dataset/images/test
  nc: 7
  names: [missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper, normal]

# 训练配置
train:
  epochs: 100
  batch: 16
  device: 0
  patience: 20
  save_period: 10
  
  # 优化器
  optimizer: AdamW
  lr0: 0.001
  lrf: 0.01
  weight_decay: 0.0005
  warmup_epochs: 3
  
  # 学习率调度
  scheduler: cosine
  warmup_momentum: 0.8
  warmup_bias_lr: 0.1

# 数据增强 (强增强)
augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 15
  translate: 0.1
  scale: 0.5
  shear: 0.0
  perspective: 0.0
  flipud: 0.0
  fliplr: 0.5
  mosaic: 1.0
  mixup: 0.15
  copy_paste: 0.1

# 导出配置
export:
  format: onnx
  opset: 12
  simplify: true
  half: false
```

### 7.2 性能指标目标

| 指标 | 当前 | 目标 |
|-----|------|------|
| mAP@50 | ~0.85 | ≥0.90 |
| mAP@50-95 | ~0.65 | ≥0.75 |
| 推理速度 (RTX 3080) | ~5ms | ≤3ms |
| 模型大小 | ~22MB | ≤15MB |

---

## 八、总结

### 架构选型建议

```
┌─────────────────────────────────────────────────────────────────┐
│                     PCB 缺陷检测技术选型总结                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  🥇 首选方案: YOLOv8-Seg                                         │
│     • 检测+分割一体化                                             │
│     • 高精度 + 实时速度                                           │
│     • 成熟生态，易部署                                            │
│                                                                  │
│  🥈 备选方案: RT-DETR                                             │
│     • 精度更高，适合高端检测                                       │
│     • 推理速度稍慢                                                │
│                                                                  │
│  🥉 边缘部署: YOLOv8n + TensorRT                                 │
│     • 模型最小 6.3MB                                              │
│     • 推理 <3ms                                                   │
│     • 适合 Jetson/Xavier                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 后续优化方向

1. **精度提升**: 引入 SAM 辅助标注 + 级联结构
2. **速度优化**: TensorRT INT8 量化
3. **泛化增强**: 域适应训练 + 仿真数据
4. **多任务**: 统一检测+分割+关键点的多任务框架
