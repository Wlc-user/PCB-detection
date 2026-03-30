"""
PCB缺陷检测 Demo 演示脚本
用于展示技术能力
"""

import cv2
import os
from pathlib import Path

# 输出目录
DEMO_DIR = Path("demo_output")
DEMO_DIR.mkdir(exist_ok=True)


def demo_single_image():
    """单图检测演示"""
    from ultralytics import YOLO
    
    model = YOLO('models/yolov8/train/weights/best.pt')
    
    # 测试图片
    test_images = [
        'yolo_pcb_dataset/images/test/01_missing_hole_02.jpg',
        'yolo_pcb_dataset/images/test/01_mouse_bite_02.jpg',
    ]
    
    for img_path in test_images:
        if Path(img_path).exists():
            print(f"检测: {img_path}")
            results = model.predict(
                img_path, 
                conf=0.25, 
                save=True,
                project=str(DEMO_DIR),
                name='single'
            )
            print(f"  结果已保存到: {DEMO_DIR}/single/")
    
    print("\n单图检测完成！")


def demo_batch():
    """批量检测演示"""
    from ultralytics import YOLO
    
    model = YOLO('models/yolov8/train/weights/best.pt')
    
    # 批量检测测试集
    print("批量检测中...")
    results = model.predict(
        'yolo_pcb_dataset/images/test',
        conf=0.25,
        save=True,
        project=str(DEMO_DIR),
        name='batch'
    )
    
    print(f"批量检测完成！检测了 {len(results)} 张图片")
    print(f"结果保存在: {DEMO_DIR}/batch/")


def demo_api():
    """API服务演示"""
    print("启动API服务...")
    print("运行: python defect_detection_api.py")
    print("然后访问: http://localhost:8000/docs")


def demo_template_matching():
    """模板对比法演示"""
    print("运行模板对比...")
    print("运行: python smart_label_v3.py")


def main():
    """主演示流程"""
    print("=" * 60)
    print("       PCB缺陷智能检测系统 - Demo演示")
    print("=" * 60)
    print()
    print("请选择演示内容:")
    print()
    print("  [1] 单图检测演示")
    print("  [2] 批量检测演示") 
    print("  [3] API服务演示")
    print("  [4] 模板对比法演示")
    print("  [5] 运行全部演示")
    print()
    print("  [0] 退出")
    print()
    
    choice = input("请输入选项 [1-5]: ").strip()
    
    if choice == '1':
        demo_single_image()
    elif choice == '2':
        demo_batch()
    elif choice == '3':
        demo_api()
    elif choice == '4':
        demo_template_matching()
    elif choice == '5':
        demo_single_image()
        demo_batch()
        demo_api()
    else:
        print("退出")


if __name__ == '__main__':
    main()
