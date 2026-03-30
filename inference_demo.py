"""
PCB缺陷检测推理演示脚本
快速上手YOLOv8模型推理
"""

import cv2
from pathlib import Path
from ultralytics import YOLO


def detect_defects(image_path, model_path='models/yolov8/train/weights/best.pt', conf=0.25):
    """
    检测PCB图片中的缺陷
    
    Args:
        image_path: 图片路径
        model_path: 模型路径
        conf: 置信度阈值
    
    Returns:
        检测结果
    """
    # 加载模型
    model = YOLO(model_path)
    
    # 推理
    results = model.predict(
        image_path, 
        conf=conf, 
        iou=0.45,
        save=True,           # 保存标注图
        save_txt=True,      # 保存标签
        project='runs/detect',
        name='predict'
    )
    
    return results


def detect_batch(image_dir, model_path='models/yolov8/train/weights/best.pt'):
    """
    批量检测文件夹中的所有图片
    
    Args:
        image_dir: 图片文件夹路径
        model_path: 模型路径
    """
    model = YOLO(model_path)
    
    # 批量推理
    results = model.predict(
        image_dir,
        conf=0.25,
        iou=0.45,
        save=True,
        project='runs/detect',
        name='batch'
    )
    
    return results


def print_results(results):
    """打印检测结果"""
    for r in results:
        boxes = r.boxes
        print(f"\n检测到 {len(boxes)} 个缺陷:")
        
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            # 类别名称
            class_names = [
                'missing_hole', 'mouse_bite', 'open_circuit',
                'short', 'spur', 'spurious_copper'
            ]
            cls_name = class_names[cls] if cls < len(class_names) else f'class_{cls}'
            
            print(f"  [{i+1}] {cls_name:18s} 置信度: {conf:.2f}  位置: {xyxy[:4]}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PCB缺陷检测推理')
    parser.add_argument('--image', type=str, help='单张图片路径')
    parser.add_argument('--dir', type=str, help='图片文件夹路径')
    parser.add_argument('--model', type=str, default='models/yolov8/train/weights/best.pt', help='模型路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    
    args = parser.parse_args()
    
    if args.image:
        # 单图检测
        print(f"检测图片: {args.image}")
        results = detect_defects(args.image, args.model, args.conf)
        print_results(results)
        
    elif args.dir:
        # 批量检测
        print(f"批量检测文件夹: {args.dir}")
        results = detect_batch(args.dir, args.model)
        print(f"完成! 检测了 {len(results)} 张图片")
        
    else:
        # 默认：检测测试集示例
        print("使用默认测试图片...")
        test_img = 'yolo_pcb_dataset/images/test/00_missing_hole_01.jpg'
        if Path(test_img).exists():
            results = detect_defects(test_img, args.model, args.conf)
            print_results(results)
        else:
            print("测试图片不存在，请使用 --image 指定图片路径")
