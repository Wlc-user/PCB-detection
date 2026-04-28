"""
PCB缺陷检测 - 平衡优化训练 v5
解决类别不平衡和小目标问题
"""

from ultralytics import YOLO
import yaml
from pathlib import Path
import torch

# 配置 - 使用平衡的数据配置
DATA_CONFIG = {
    'path': str(Path('yolo_pcb_dataset').absolute()),
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    'names': ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper'],
    'nc': 6
}

Path('models/yolov8').mkdir(parents=True, exist_ok=True)
with open('models/yolov8/data.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(DATA_CONFIG, f, allow_unicode=True)


def train_balanced():
    """平衡优化训练 - 重新从头训练"""
    print("\n" + "="*60)
    print("开始平衡优化训练 v5")
    print("="*60)
    
    # 重新从头训练，使用本地yolov8n.pt
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data='models/yolov8/data.yaml',
        
        # ========== 训练配置 ==========
        epochs=30
        ,
        imgsz=1280,
        batch=4,
        patience=80,
        
        name='train_balanced',
        project='models/yolov8',
        exist_ok=True,
        
        # ========== 优化器 (更稳定) ==========
        optimizer='AdamW',
        lr0=0.0002,        # 更小的初始学习率
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,   # 更长warmup
        warmup_bias_lr=0.1,
        warmup_momentum=0.8,
        
        # ========== 损失函数 ==========
        box=7.5,
        cls=3.0,           # 大幅增加分类损失
        dfl=1.5,
        
        # ========== 设备 ==========
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=0,
        amp=True,
        
        # ========== 数据增强 (平衡策略) ==========
        augment=True,
        mosaic=1.0,
        mixup=0.3,
        copy_paste=0.3,
        
        # 几何增强
        degrees=10.0,
        translate=0.15,
        scale=0.5,
        shear=1.0,
        perspective=0.0005,
        
        # 颜色增强
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        
        # 翻转
        flipud=0.0,
        fliplr=0.5,
        
        # ========== 验证 ==========
        val=True,
        plots=True,
        
        # ========== 推理配置 ==========
        conf=0.1,
        iou=0.5,
        
        verbose=True,
    )
    
    print("\n训练完成!")


if __name__ == '__main__':
    train_balanced()
