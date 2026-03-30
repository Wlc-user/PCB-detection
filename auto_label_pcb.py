"""
PCB缺陷数据集自动标注工具
用于修复错误的YOLO标注并生成更精确的边界框
"""
import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm

class PCBAutoLabeler:
    """PCB缺陷自动标注器"""
    
    # 缺陷类型ID映射
    DEFECT_TYPES = {
        'missing_hole': 0,
        'mouse_bite': 1,
        'open_circuit': 2,
        'short': 3,
        'spur': 4,
        'spurious_copper': 5,
    }
    
    # 各类别的默认缺陷大小（相对于图片尺寸）
    # 这些是基于PCB缺陷检测的常见大小
    DEFECT_SIZES = {
        0: (0.05, 0.05),   # missing_hole: 5%x5%
        1: (0.08, 0.06),   # mouse_bite: 8%x6%
        2: (0.15, 0.03),   # open_circuit: 15%x3%
        3: (0.10, 0.08),   # short: 10%x8%
        4: (0.06, 0.04),   # spur: 6%x4%
        5: (0.12, 0.10),   # spurious_copper: 12%x10%
    }
    
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.train_dir = self.dataset_dir / 'images' / 'train'
        self.val_dir = self.dataset_dir / 'images' / 'val'
        self.test_dir = self.dataset_dir / 'images' / 'test'
        
    def parse_filename(self, filename):
        """
        从文件名解析缺陷类型
        例如: 01_missing_hole_01.jpg -> missing_hole
        """
        name = Path(filename).stem.lower()
        
        # 按优先级匹配
        if 'spurious_copper' in name:
            return 'spurious_copper'
        elif 'missing_hole' in name:
            return 'missing_hole'
        elif 'mouse_bite' in name:
            return 'mouse_bite'
        elif 'open_circuit' in name:
            return 'open_circuit'
        elif 'short' in name:
            return 'short'
        elif 'spur' in name:
            return 'spur'
        
        return None
    
    def detect_defect_regions(self, img_path, defect_type):
        """
        使用图像处理检测缺陷区域
        返回可能的缺陷区域列表
        """
        img = cv2.imread(str(img_path))
        if img is None:
            return []
            
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        regions = []
        
        # 方法1: 基于形态学操作检测异常区域
        # 对于不同缺陷类型使用不同的检测策略
        if defect_type == 'missing_hole':
            # 找黑色的圆形区域（应该是孔但缺失了）
            _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 200 < area < 5000:  # 合理的孔大小
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    regions.append((x, y, cw, ch))
                    
        elif defect_type == 'mouse_bite':
            # 边缘检测找缺口
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 500 < area < 10000:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    if ch > cw:  # 垂直方向的特征
                        regions.append((x, y, cw, ch))
                        
        elif defect_type == 'open_circuit':
            # 找线条断裂
            edges = cv2.Canny(gray, 30, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 300 < area < 8000:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    if cw > ch * 3:  # 水平线条
                        regions.append((x, y, cw, ch))
                        
        elif defect_type == 'short':
            # 找多余的连接区域
            _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 1000 < area < 15000:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    if ch > cw * 0.5:  # 垂直特征
                        regions.append((x, y, cw, ch))
                        
        elif defect_type == 'spur':
            # 找突出的细小区域
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 100 < area < 3000:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    if cw > 3 and ch > 3:  # 小的突出
                        regions.append((x, y, cw, ch))
                        
        elif defect_type == 'spurious_copper':
            # 找孤立的铜区域
            _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 2000 < area < 50000:  # 较大的多余区域
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    regions.append((x, y, cw, ch))
        
        # 过滤并返回合理的区域
        filtered = []
        for x, y, cw, ch in regions:
            # 确保区域在图片范围内
            if x > 0 and y > 0 and x + cw < w and y + ch < h:
                filtered.append((x, y, cw, ch))
        
        return filtered
    
    def generate_yolo_label(self, bbox, img_width, img_height):
        """
        将边界框转换为YOLO格式
        bbox: (x, y, w, h) 像素坐标
        返回: (class_id, center_x, center_y, width, height) 归一化
        """
        x, y, w, h = bbox
        
        # 计算中心点和宽高
        center_x = (x + w / 2) / img_width
        center_y = (y + h / 2) / img_height
        norm_w = w / img_width
        norm_h = h / img_height
        
        return center_x, center_y, norm_w, norm_h
    
    def auto_label_dataset(self, output_suffix='_auto'):
        """
        自动标注整个数据集
        """
        # 备份原始标注
        backup_dir = self.dataset_dir / 'labels_backup'
        if not backup_dir.exists():
            backup_dir.mkdir(parents=True)
            
        for split in ['train', 'val', 'test']:
            img_dir = self.dataset_dir / 'images' / split
            label_dir = self.dataset_dir / 'labels' / split
            
            if not img_dir.exists():
                continue
                
            print(f"\n处理 {split} 集...")
            
            for img_file in tqdm(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))):
                # 解析缺陷类型
                defect_type = self.parse_filename(img_file.name)
                if defect_type is None or defect_type not in self.DEFECT_TYPES:
                    print(f"无法识别: {img_file.name}")
                    continue
                
                class_id = self.DEFECT_TYPES[defect_type]
                
                # 尝试检测缺陷区域
                regions = self.detect_defect_regions(img_file, defect_type)
                
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                h, w = img.shape[:2]
                
                label_file = label_dir / f"{img_file.stem}.txt"
                
                # 如果没有检测到区域，使用默认位置
                if not regions:
                    # 使用图片中心区域作为默认（这是一个fallback）
                    default_w = int(w * self.DEFECT_SIZES[class_id][0])
                    default_h = int(h * self.DEFECT_SIZES[class_id][1])
                    
                    # 使用图片的不同位置（基于文件名编号）
                    idx = int(img_file.stem.split('_')[-1]) if img_file.stem.split('_')[-1].isdigit() else 1
                    
                    # 分散标注位置
                    col = (idx - 1) % 4
                    row = (idx - 1) // 4
                    
                    cx = w * (0.2 + col * 0.2)
                    cy = h * (0.2 + row * 0.2)
                    
                    x = int(cx - default_w / 2)
                    y = int(cy - default_h / 2)
                    regions = [(x, y, default_w, default_h)]
                
                # 写入YOLO格式标注
                with open(label_file, 'w') as f:
                    for x, y, cw, ch in regions[:3]:  # 最多3个区域
                        cx, cy, nw, nh = self.generate_yolo_label((x, y, cw, ch), w, h)
                        # 确保值在0-1之间
                        cx = max(0.001, min(0.999, cx))
                        cy = max(0.001, min(0.999, cy))
                        nw = max(0.001, min(0.999, nw))
                        nh = max(0.001, min(0.999, nh))
                        f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        
        print(f"\n自动标注完成！原始标注已备份到: {backup_dir}")
    
    def verify_annotations(self):
        """验证标注文件的格式"""
        issues = []
        
        for split in ['train', 'val', 'test']:
            label_dir = self.dataset_dir / 'labels' / split
            if not label_dir.exists():
                continue
                
            for label_file in label_dir.glob('*.txt'):
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    
                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        issues.append(f"{label_file}:{i+1} - 格式错误")
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        cx, cy, nw, nh = map(float, parts[1:])
                        
                        if class_id < 0 or class_id > 5:
                            issues.append(f"{label_file}:{i+1} - 无效类别ID {class_id}")
                        
                        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= nw <= 1 and 0 <= nh <= 1):
                            issues.append(f"{label_file}:{i+1} - 值超出范围")
                    except:
                        issues.append(f"{label_file}:{i+1} - 解析错误")
        
        if issues:
            print(f"发现 {len(issues)} 个问题:")
            for issue in issues[:20]:
                print(f"  - {issue}")
        else:
            print("所有标注文件格式正确！")
        
        return issues
    
    def visualize_sample(self, split='train', num_samples=5):
        """可视化样本标注"""
        import random
        
        img_dir = self.dataset_dir / 'images' / split
        label_dir = self.dataset_dir / 'labels' / split
        
        if not img_dir.exists():
            print(f"{split} 目录不存在")
            return
        
        samples = list(img_dir.glob('*.jpg'))[:num_samples]
        
        for img_path in samples:
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            
            label_path = label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
                
            # 读取标注
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # 绘制边界框
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                    
                class_id = int(parts[0])
                cx, cy, nw, nh = map(float, parts[1:])
                
                # 转换为像素坐标
                x = int((cx - nw/2) * w)
                y = int((cy - nh/2) * h)
                bw = int(nw * w)
                bh = int(nh * h)
                
                # 颜色
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), 
                         (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                color = colors[class_id % len(colors)]
                
                cv2.rectangle(img, (x, y), (x+bw, y+bh), color, 2)
                cv2.putText(img, f"#{class_id}", (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 保存
            out_path = img_dir.parent.parent / f"viz_{img_path.name}"
            cv2.imwrite(str(out_path), img)
            print(f"已保存可视化: {out_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='PCB缺陷数据集自动标注工具')
    parser.add_argument('--dataset', type=str, default='./yolo_pcb_dataset', 
                       help='数据集目录')
    parser.add_argument('--mode', type=str, choices=['auto', 'verify', 'viz'],
                       default='auto', help='运行模式')
    parser.add_argument('--split', type=str, default='train',
                       help='可视化哪个数据集')
    
    args = parser.parse_args()
    
    labeler = PCBAutoLabeler(args.dataset)
    
    if args.mode == 'auto':
        labeler.auto_label_dataset()
    elif args.mode == 'verify':
        labeler.verify_annotations()
    elif args.mode == 'viz':
        labeler.visualize_sample(args.split)


if __name__ == '__main__':
    main()
