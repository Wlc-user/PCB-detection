"""
DeepPCB 数据集下载和转换脚本

使用方法:
1. 手动下载 DeepPCB 数据集: https://github.com/tangsanli5201/DeepPCB
2. 解压到 DeepPCB 目录
3. 运行: python convert_deepcb.py
"""

import os
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

# DeepPCB 类别映射
DEFECT_CLASSES = {
    'missing_hole': 0,
    'mouse_bite': 1,
    'open_circuit': 2,
    'short': 3,
    'spur': 4,
    'spurious_copper': 5,
}

def convert_annotations(xml_dir, output_dir, image_dir):
    """转换 XML 标注为 YOLO 格式"""
    os.makedirs(output_dir, exist_ok=True)
    
    for xml_file in Path(xml_dir).glob('*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 获取图片尺寸
        size = root.find('size')
        img_w = int(size.find('width').text)
        img_h = int(size.find('height').text)
        
        # 获取文件名
        img_name = root.find('filename').text
        label_name = Path(img_name).stem + '.txt'
        
        yolo_labels = []
        
        for obj in root.findall('object'):
            cls_name = obj.find('name').text.lower()
            if cls_name not in DEFECT_CLASSES:
                continue
            
            cls_id = DEFECT_CLASSES[cls_name]
            
            # 获取边界框
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # 转换为 YOLO 格式 (中心点 + 宽高, 归一化)
            cx = (xmin + xmax) / 2 / img_w
            cy = (ymin + ymax) / 2 / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h
            
            yolo_labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        
        # 保存标注文件
        if yolo_labels:
            with open(os.path.join(output_dir, label_name), 'w') as f:
                f.write('\n'.join(yolo_labels))
            
            # 复制图片
            src_img = os.path.join(image_dir, img_name)
            if os.path.exists(src_img):
                dst_img = os.path.join(output_dir.replace('labels', 'images'), img_name)
                os.makedirs(os.path.dirname(dst_img), exist_ok=True)
                if not os.path.exists(dst_img):
                    shutil.copy2(src_img, dst_img)

def main():
    base_dir = Path('DeepPCB')
    
    if not base_dir.exists():
        print("错误: 请先下载 DeepPCB 数据集")
        print("=" * 50)
        print("下载方式:")
        print("1. 访问: https://github.com/tangsanli5201/DeepPCB")
        print("2. 下载 ZIP 文件")
        print("3. 解压到项目根目录")
        print("4. 重新运行此脚本")
        print("=" * 50)
        return
    
    print("开始转换 DeepPCB 数据集...")
    
    # 创建输出目录
    yolo_dir = Path('yolo_pcb_dataset_deepcb')
    for split in ['train', 'test']:
        (yolo_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 转换训练集
    if (base_dir / 'train').exists():
        print("转换训练集...")
        convert_annotations(
            base_dir / 'train' / 'xml',
            yolo_dir / 'labels' / 'train',
            base_dir / 'train' / 'jpg'
        )
    
    # 转换测试集
    if (base_dir / 'test').exists():
        print("转换测试集...")
        convert_annotations(
            base_dir / 'test' / 'xml',
            yolo_dir / 'labels' / 'test',
            base_dir / 'test' / 'jpg'
        )
    
    print(f"\n转换完成! 数据保存在: {yolo_dir}")
    
    # 统计
    train_imgs = len(list((yolo_dir / 'images' / 'train').glob('*.jpg')))
    test_imgs = len(list((yolo_dir / 'images' / 'test').glob('*.jpg')))
    print(f"训练集: {train_imgs} 张图片")
    print(f"测试集: {test_imgs} 张图片")

if __name__ == '__main__':
    main()
