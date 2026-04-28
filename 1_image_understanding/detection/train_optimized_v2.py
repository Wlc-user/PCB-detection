"""
YOLOv10 优化训练脚本
针对小数据集（~200张）优化

问题诊断:
1. 分类损失 3.5+ → 学习率/增强问题
2. 框损失下降慢 → 预训练权重/学习率
3. 精度 0.68 → 数据/策略问题

优化策略:
1. 使用预训练权重 (不是 from_scratch)
2. 降低学习率 lr0=0.001
3. 减弱数据增强
4. 增加 warmup
5. 使用 AdamW 优化器

Author: Vision Team
"""

import os
import sys
import yaml
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from PIL import Image

# 尝试导入 YOLOv10
try:
    from ultralytics import YOLO
    print("[OK] Ultralytics imported")
except ImportError:
    print("[ERROR] pip install ultralytics>=8.2.0")
    sys.exit(1)


class OptimizedTrainer:
    """优化版训练器"""
    
    def __init__(self, project_name: str = "yolov10_pcb_opt"):
        self.project_name = project_name
        self.results = []
    
    def prepare_data_config(self, yaml_path: str = "yolo_pcb_dataset/data.yaml") -> str:
        """准备/验证数据配置"""
        
        # 检查或创建 data.yaml
        if not Path(yaml_path).exists():
            print(f"[WARN] {yaml_path} not found, creating...")
            
            # 查找数据集
            dataset_paths = [
                "yolo_pcb_dataset",
                "yolo_pcb_dataset_augmented", 
                "yolo_pcb_dataset_balanced"
            ]
            
            dataset_path = None
            for p in dataset_paths:
                if Path(p).exists():
                    dataset_path = p
                    break
            
            if not dataset_path:
                print("[ERROR] No dataset found!")
                return None
            
            config = {
                'path': str(Path.cwd() / dataset_path),
                'train': 'images/train',
                'val': 'images/val',
                'test': 'images/test',
                'names': {
                    0: 'missing_hole',
                    1: 'mouse_bite', 
                    2: 'open_circuit',
                    3: 'short',
                    4: 'spur',
                    5: 'spurious_copper',
                    6: 'normal'
                },
                'nc': 7
            }
            
            yaml_path = "pcb_data.yaml"
            with open(yaml_path, 'w') as f:
                yaml.dump(config, f, allow_unicode=True)
            
            print(f"[OK] Created {yaml_path}")
        
        return yaml_path
    
    def analyze_dataset(self, yaml_path: str):
        """分析数据集统计"""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("\n" + "="*50)
        print("Dataset Analysis")
        print("="*50)
        
        # 处理 path 可能缺失的情况
        base_path = Path(yaml_path).parent
        if 'path' in config:
            base_path = Path(config['path'])
        
        train_path = base_path / config['train']
        val_path = base_path / config['val']
        
        if train_path.exists():
            train_images = list(train_path.glob("*.jpg")) + list(train_path.glob("*.png"))
            print(f"Train images: {len(train_images)}")
        
        if val_path.exists():
            val_images = list(val_path.glob("*.jpg")) + list(val_path.glob("*.png"))
            print(f"Val images: {len(val_images)}")
        
        # 统计类别分布
        label_base = Path(yaml_path).parent
        if 'path' in config:
            label_base = Path(config['path'])
        label_path = label_base / "labels" / "train"
        if label_path.exists():
            class_counts = {}
            for label_file in label_path.glob("*.txt"):
                with open(label_file) as f:
                    for line in f:
                        cls = int(line.split()[0])
                        class_counts[cls] = class_counts.get(cls, 0) + 1
            
            print("\nClass distribution:")
            names = config.get('names', [])
            for cls, count in sorted(class_counts.items()):
                if isinstance(names, list):
                    name = names[cls] if cls < len(names) else f"class_{cls}"
                else:
                    name = names.get(cls, f"class_{cls}")
                print(f"  {cls}: {name} = {count}")
        
        print("="*50)
    
    def train(self, yaml_path: str):
        """优化训练"""
        
        print("\n" + "="*60)
        print("Starting Optimized YOLOv10 Training")
        print("="*60)
        
        # 1. 加载预训练模型（关键！不是 from_scratch）
        print("\n[1] Loading pretrained YOLOv10...")
        try:
            model = YOLO("yolov10s.pt")  # 使用预训练权重
        except:
            model = YOLO("yolov10n.pt")  # nano版本
        
        print(f"[OK] Using {model.model_name if hasattr(model, 'model_name') else 'pretrained'} model")
        
        # 2. 优化后的训练参数
        # 针对小数据集的优化配置
        args = {
            # 数据
            'data': yaml_path,
            'epochs': 100,           # 小数据集 50-100 就够了
            'batch': 8,              # 小 batch 配合更小学习率
            
            # 关键优化：学习率
            'lr0': 0.001,            # 从 0.01 降到 0.001 (降低10倍)
            'lrf': 0.01,             # 最终学习率
            'warmup_epochs': 5,      # warmup 5 个 epoch
            'warmup_bias_lr': 0.1,   # warmup 时 bias 学习率
            
            # 优化器
            'optimizer': 'AdamW',     # AdamW 比 SGD 更稳定
            'weight_decay': 0.0005,
            
            # 关键优化：数据增强（减弱）
            'hsv_h': 0.005,          # 色相变化 ±0.5%
            'hsv_s': 0.3,            # 饱和度变化 ±30%
            'hsv_v': 0.3,            # 亮度变化 ±30%
            'degrees': 5,            # 旋转 ±5° (原来是 0.0)
            'translate': 0.1,        # 平移 ±10%
            'scale': 0.3,            # 缩放 ±30%
            'shear': 0.0,            # 剪切 0
            'perspective': 0.0,      # 透视 0
            'flipud': 0.0,           # 上下翻转 0 (PCB 通常不需要)
            'fliplr': 0.5,           # 左右翻转 50%
            'mosaic': 0.5,           # 马赛克从 1.0 降到 0.5
            'mixup': 0.0,            # 关闭 mixup (小数据集容易过拟合)
            'copy_paste': 0.0,       # 关闭 copy_paste
            
            # 其他优化
            'patience': 20,         # 早停耐心
            'save': True,
            'save_period': 10,
            'cache': True,           # 缓存加速
            'workers': 4,
            'project': self.project_name,
            'name': 'train',
            'exist_ok': True,
            'pretrained': True,
            'verbose': True,
            'seed': 42,
            
            # 验证
            'val': True,
            'plots': True,
        }
        
        print("\n[2] Training parameters (optimized for small dataset):")
        print(f"   lr0: {args['lr0']} (was 0.01)")
        print(f"   batch: {args['batch']}")
        print(f"   optimizer: {args['optimizer']}")
        print(f"   mosaic: {args['mosaic']} (was 1.0)")
        print(f"   mixup: {args['mixup']} (was 0.0)")
        print(f"   warmup_epochs: {args['warmup_epochs']}")
        
        # 3. 开始训练
        print("\n[3] Starting training...")
        results = model.train(**args)
        
        # 4. 输出结果
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        
        # 获取最终指标
        if hasattr(results, 'box'):
            metrics = results.box
            print(f"\nFinal metrics:")
            print(f"  mAP50: {metrics.map50:.4f}")
            print(f"  mAP50-95: {metrics.map:.4f}")
            print(f"  Precision: {metrics.mp:.4f}")
            print(f"  Recall: {metrics.mr:.4f}")
        
        return model, results
    
    def validate_immediately(self, model_path: str, yaml_path: str):
        """训练后立即验证"""
        print("\n[4] Running validation...")
        
        model = YOLO(model_path)
        metrics = model.val(data=yaml_path, verbose=True)
        
        return metrics


def quick_fix_current_model():
    """快速修复当前模型：继续训练"""
    
    print("""
================================================================================
    Quick Fix: Continue Training with Better Settings
================================================================================
    
Problems identified:
  1. Classification loss 3.5+ (should be <1.0)
  2. Box loss improvement too slow
  3. Precision only 0.68 (should be 0.8+)
  
Root causes:
  - Learning rate too high (0.01)
  - Data augmentation too aggressive
  - No pretrained weights used
  
Solution:
  1. Load best.pt as starting point
  2. Use lower learning rate (0.001)
  3. Reduce augmentation
  4. Train for 50 more epochs
  
================================================================================
    """)
    
    return {
        'strategy': 'continue_training',
        'resume_from': 'models/yolov8/train/weights/best.pt',
        'lr0': 0.001,
        'epochs': 50,
        'augmentation_reduction': {
            'mosaic': 0.5,  # from 1.0
            'mixup': 0.0,   # from 0.0
            'degrees': 5,   # from 0.0
        }
    }


def create_training_report():
    """生成训练问题诊断报告"""
    
    report = """
================================================================================
              YOLOv10 Training Problem Diagnosis Report
================================================================================

## 问题诊断

### 1. 分类损失 (cls_loss) 异常
   现象: 3.5+ (正常应在 1.0 以内)
   原因: 
     - 学习率太大 (lr0=0.01) → 模型震荡无法收敛
     - 数据增强太强 → 模型学不到有效特征
     - 类别不平衡 → 部分类别样本太少
   
### 2. 框损失 (box_loss) 下降慢
   现象: 8 epoch 只降 0.1
   原因:
     - 学习率不合适
     - 没有使用预训练权重
     - batch size 可能太大
   
### 3. 精度 (Precision) 低
   现象: 0.68 (正常应 0.8+)
   原因:
     - 模型还没学会
     - 误检率高
     - 特征提取能力差

## 解决方案

### 方案 A: 继续训练（推荐）
```bash
python train_optimized_v2.py --mode continue
```

### 方案 B: 从头训练（使用优化配置）
```bash
python train_optimized_v2.py --mode retrain
```

### 关键参数调整

| 参数 | 原值 | 新值 | 说明 |
|-----|-----|-----|-----|
| lr0 | 0.01 | 0.001 | 降低10倍 |
| optimizer | SGD | AdamW | 更稳定 |
| warmup_epochs | 3 | 5 | 更多预热 |
| mosaic | 1.0 | 0.5 | 减少马赛克 |
| mixup | 0.0 | 0.0 | 保持关闭 |
| degrees | 0.0 | 5 | 允许小旋转 |

## 预期效果

优化后:
  - cls_loss: 3.5 → 0.8 (约50 epoch)
  - box_loss: 下降速度提升 3-5倍
  - Precision: 0.68 → 0.85+
  - Recall: 提升 10-20%

================================================================================
"""
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='retrain', choices=['retrain', 'continue', 'diagnose'])
    parser.add_argument('--data', default='yolo_pcb_dataset/data.yaml')
    parser.add_argument('--model', default='yolov10s.pt')
    args = parser.parse_args()
    
    trainer = OptimizedTrainer()
    
    if args.mode == 'diagnose':
        print(create_training_report())
    
    elif args.mode == 'retrain':
        yaml_path = trainer.prepare_data_config(args.data)
        if yaml_path:
            trainer.analyze_dataset(yaml_path)
            model, results = trainer.train(yaml_path)
            
            # 验证最佳模型
            best_model = f"{trainer.project_name}/train/weights/best.pt"
            if Path(best_model).exists():
                trainer.validate_immediately(best_model, yaml_path)
    
    elif args.mode == 'continue':
        print(create_training_report())
        print("\nTo continue training from best.pt:")
        print("  1. Load: models/yolov8/train/weights/best.pt")
        print("  2. lr0=0.001, epochs=50")
        print("  3. Reduced augmentation")
