"""
PCB特征工程测试 - 验证六种缺陷模式匹配
"""

from ultralytics import YOLO
from pcb_features import PCBFeatureExtractor, PCBMultiFeatureDetector, DefectType
import cv2
import numpy as np
from pathlib import Path


def test_feature_extraction():
    """测试特征提取"""
    print("=" * 60)
    print("测试 PCB 特征提取器")
    print("=" * 60)
    
    extractor = PCBFeatureExtractor()
    
    # 打印六种缺陷的特征模板
    print("\n六种缺陷特征模板:")
    print("-" * 60)
    
    for defect_type in DefectType:
        template = extractor.DEFECT_TEMPLATES[defect_type]
        print(f"\n【{defect_type.value}】")
        for feature, (min_v, max_v) in template.items():
            print(f"  {feature:15}: {min_v:8.2f} ~ {max_v:8.2f}")


def test_on_images():
    """在真实图像上测试"""
    print("\n" + "=" * 60)
    print("在真实图像上测试")
    print("=" * 60)
    
    # 加载YOLO模型
    model_path = 'models/yolov8/train/weights/best.pt'
    model = YOLO(model_path)
    
    # 创建融合检测器
    detector = PCBMultiFeatureDetector(yolo_model=model)
    
    # 测试图像
    test_dir = Path('yolo_pcb_dataset/images/test')
    if not test_dir.exists():
        test_dir = Path('yolo_pcb_dataset/images/val')
    
    image_files = list(test_dir.glob('*.jpg'))[:3]
    
    if not image_files:
        print("未找到测试图像")
        return
    
    for img_path in image_files:
        print(f"\n测试图像: {img_path.name}")
        print("-" * 40)
        
        image = cv2.imread(str(img_path))
        if image is None:
            print("  读取失败")
            continue
        
        # 综合检测
        results = detector.detect_from_image(image, conf=0.15)
        
        print(f"  深度学习检测: {len(results['deep_learning'])} 个")
        for d in results['deep_learning']:
            print(f"    - {d['type']}: {d['confidence']:.2f}")
        
        print(f"  特征工程检测: {len(results['feature_engineering'])} 个")
        for d in results['feature_engineering']:
            print(f"    - {d['type']}: {d['confidence']:.2f}")
        
        print(f"  融合结果: {len(results['fusion'])} 个")
        for d in results['fusion']:
            print(f"    - {d['type']}: {d['confidence']:.2f}")


def visualize_defect_patterns():
    """可视化六种缺陷的典型特征"""
    print("\n" + "=" * 60)
    print("六种缺陷典型特征模式")
    print("=" * 60)
    
    patterns = {
        'missing_hole': {
            'desc': '圆形区域内无铜箔/过孔缺失',
            'features': ['高圆度(0.7-1.0)', '固定面积', '边缘清晰'],
            'detect': '圆形度检测 + 孔洞填充检测'
        },
        'mouse_bite': {
            'desc': '导线边缘被侵蚀，形成缺口',
            'features': ['不规则边缘', '低圆度(0.3-0.7)', '低填充率'],
            'detect': '轮廓凹凸度检测 + 边缘断裂分析'
        },
        'open_circuit': {
            'desc': '导线断裂，电流中断',
            'features': ['细长形状', '极高长宽比(3-20)', '低圆度'],
            'detect': '线段断裂检测 + 端点分析'
        },
        'short': {
            'desc': '相邻导线异常连接',
            'features': ['不规则大面积', '低圆度', '高填充率'],
            'detect': '连通域分析 + 间距检测'
        },
        'spur': {
            'desc': '导线边缘突出的小刺状物',
            'features': ['细小狭长', '高长宽比(2-8)', '低圆度'],
            'detect': '边缘毛刺检测 + 尖角分析'
        },
        'spurious_copper': {
            'desc': '非设计区域存在多余铜箔',
            'features': ['大面积', '不规则形状', '高填充率'],
            'detect': '模板比对 + DRC规则检查'
        }
    }
    
    for defect, info in patterns.items():
        print(f"\n【{defect}】")
        print(f"  描述: {info['desc']}")
        print(f"  特征: {', '.join(info['features'])}")
        print(f"  检测: {info['detect']}")


if __name__ == '__main__':
    test_feature_extraction()
    visualize_defect_patterns()
    test_on_images()
