"""
PCB缺陷检测 - 重新开始训练 v6
解决欠拟合问题，从更强的基础模型开始
"""

from ultralytics import YOLO
import yaml
from pathlib import Path
import torch

# 配置
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


def train_with_pretrained():
    """使用预训练模型继续训练"""
    print("="*60)
    print("重新开始训练 v6 - 使用预训练权重")
    print("="*60)
    
    # 使用现有的 best.pt 作为基础（它已经有一定的知识）
    model_path = 'models/yolov8/train/weights/best.pt'
    
    if not Path(model_path).exists():
        print(f"错误: 找不到 {model_path}")
        print("请确保有训练好的模型")
        return
    
    print(f"基础模型: {model_path}")
    model = YOLO(model_path)
    
    # 评估当前性能
    print("\n[当前模型性能评估]")
    metrics = model.val(data='models/yolov8/data.yaml', conf=0.2, verbose=False, workers=0)
    print(f"  P={metrics.box.mp:.3f}, R={metrics.box.mr:.3f}, mAP50={metrics.box.map50:.3f}")
    
    # 继续训练
    print("\n[开始微调训练]")
    results = model.train(
        data='models/yolov8/data.yaml',
        
        # 训练配置
        epochs=100,
        imgsz=1280,
        batch=4,
        patience=30,
        
        name='train_v6_finetune',
        project='models/yolov8',
        exist_ok=True,
        
        # 优化器 - 小学习率微调
        optimizer='AdamW',
        lr0=0.0001,        # 很小，微调
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        
        # 损失函数
        box=7.0,
        cls=2.0,
        dfl=1.5,
        
        # 设备
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=0,
        amp=True,
        
        # 数据增强 - 适度
        augment=True,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.2,
        
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=1.0,
        
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        
        flipud=0.0,
        fliplr=0.5,
        
        val=True,
        plots=True,
        conf=0.1,
        iou=0.5,
        verbose=True,
    )
    
    # 训练后评估
    print("\n[训练完成评估]")
    model = YOLO('models/yolov8/train_v6_finetune/weights/best.pt')
    metrics = model.val(data='models/yolov8/data.yaml', conf=0.2, verbose=False, workers=0)
    print(f"  P={metrics.box.mp:.3f}, R={metrics.box.mr:.3f}, mAP50={metrics.box.map50:.3f}")
    
    return results


if __name__ == '__main__':
    train_with_pretrained()
