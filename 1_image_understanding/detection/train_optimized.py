#!/usr/bin/env python3
"""
PCB缺陷检测 - 优化训练脚本
第一优先级优化方案
"""

from ultralytics import YOLO
import yaml


def train():
    """训练优化"""
    
    # 加载配置
    with open("configs/train_config_v2.yaml") as f:
        config = yaml.safe_load(f)
    
    # 模型选择
    model_name = config["model"]["name"]  # yolov8s
    
    # 加载预训练模型
    model = YOLO(f"{model_name}.pt")
    
    # 训练
    results = model.train(
        # 数据
        data="yolo_pcb_dataset/data.yaml",
        
        # 训练参数
        epochs=config["train"]["epochs"],
        batch=config["train"]["batch"],
        imgsz=config["train"]["imgsz"],
        patience=config["train"]["patience"],
        save_period=config["train"]["save_period"],
        device=config["train"]["device"],
        
        # 损失权重 - 关键优化
        cls=config["train"]["cls"],
        box=config["train"]["box"],
        dfl=config["train"]["dfl"],
        
        # 优化器
        optimizer=config["optimizer"]["name"],
        lr0=config["optimizer"]["lr0"],
        lrf=config["optimizer"]["lrf"],
        momentum=config["optimizer"]["momentum"],
        weight_decay=config["optimizer"]["weight_decay"],
        warmup_epochs=config["optimizer"]["warmup_epochs"],
        warmup_momentum=config["optimizer"]["warmup_momentum"],
        warmup_bias_lr=config["optimizer"]["warmup_bias_lr"],
        
        # 数据增强
        hsv_h=config["augmentation"]["hsv_h"],
        hsv_s=config["augmentation"]["hsv_s"],
        hsv_v=config["augmentation"]["hsv_v"],
        degrees=config["augmentation"]["degrees"],
        translate=config["augmentation"]["translate"],
        scale=config["augmentation"]["scale"],
        shear=config["augmentation"]["shear"],
        perspective=config["augmentation"]["perspective"],
        flipud=config["augmentation"]["flipud"],
        fliplr=config["augmentation"]["fliplr"],
        mosaic=config["augmentation"]["mosaic"],
        mixup=config["augmentation"]["mixup"],
        copy_paste=config["augmentation"]["copy_paste"],
        
        # 项目
        name="yolov8s_pcb_v2",
        exist_ok=True,
        pretrained=True,
        verbose=True,
        
        # 其他
        close_mosaic=10,  # 关闭最后10轮的mosaic
    )
    
    print(f"训练完成! 保存位置: {results.save_dir}")
    return results


if __name__ == "__main__":
    train()
