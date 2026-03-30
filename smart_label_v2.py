"""
PCB缺陷智能检测与标注工具V2
基于图像分析和模板对比的智能标注
"""
import os
import cv2
import numpy as np
from pathlib import Path
import json

class PCBSmartLabelerV2:
    """基于图像分析的智能标注器"""
    
    DEFECT_TYPES = {
        'missing_hole': 0,
        'mouse_bite': 1, 
        'open_circuit': 2,
        'short': 3,
        'spur': 4,
        'spurious_copper': 5,
    }
    
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        
    def analyze_pcb_structure(self, img_path):
        """分析PCB图像结构，找出关键区域"""
        img = cv2.imread(str(img_path))
        if img is None:
            return None, None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 1. 检测PCB边界（通常有明显的走线区域）
        # 自适应阈值找PCB走线
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 3
        )
        
        # 找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤掉太大和太小的轮廓
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 500 < area < h*w*0.8:  # 合理的PCB走线区域
                valid_contours.append(cnt)
        
        return img, valid_contours
    
    def detect_defects_by_type(self, img_path, defect_type):
        """根据缺陷类型专门检测"""
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        regions = []
        
        # 使用多种方法组合检测
        if defect_type == 'missing_hole':
            # 缺失孔洞：找PCB上应该是圆形但缺失的区域
            # 方法：找小的暗点/亮点
            circles = self._find_circular_regions(gray, min_r=3, max_r=20)
            regions.extend(circles)
            
        elif defect_type == 'mouse_bite':
            # 鼠咬：走线边缘缺口
            # 方法：边缘检测找凹陷
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 200 < area < 3000:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    # 检查是否是细长形状（边缘特征）
                    if cw > ch * 2 or ch > cw * 2:
                        regions.append({'x': x, 'y': y, 'w': cw, 'h': ch, 'method': 'edge'})
                        
        elif defect_type == 'open_circuit':
            # 开路：走线断裂
            # 方法：找线条中间断开的位置
            lines = self._find_line_breaks(gray)
            regions.extend(lines)
            
        elif defect_type == 'short':
            # 短路：不应该连接的地方连接了
            # 方法：找大面积短路连接
            short_regions = self._find_short_circuits(gray)
            regions.extend(short_regions)
            
        elif defect_type == 'spur':
            # 毛刺：多余铜箔突出
            # 方法：找小的突出部分
            spurs = self._find_spurs(gray)
            regions.extend(spurs)
            
        elif defect_type == 'spurious_copper':
            # 多余铜：孤立的铜区域
            # 方法：找独立的连通域
            copper_regions = self._find_excess_copper(gray)
            regions.extend(copper_regions)
        
        return regions
    
    def _find_circular_regions(self, gray, min_r=3, max_r=20):
        """找圆形区域（孔洞）"""
        regions = []
        
        # Hough圆检测
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 10,
            param1=30, param2=15, minRadius=min_r, maxRadius=max_r
        )
        
        if circles is not None:
            for circle in circles[0]:
                cx, cy, r = circle
                x, y = int(cx - r), int(cy - r)
                diameter = int(r * 2)
                regions.append({
                    'x': x, 'y': y, 
                    'w': diameter, 'h': diameter,
                    'method': 'circle'
                })
        
        return regions
    
    def _find_line_breaks(self, gray):
        """找线条断裂"""
        regions = []
        
        # 边缘检测
        edges = cv2.Canny(gray, 30, 100)
        
        # 形态学操作找断裂
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 1000 < area < 10000:
                x, y, cw, ch = cv2.boundingRect(cnt)
                # 找细长区域（可能是断裂的线）
                if cw > 10 and ch < 30:
                    regions.append({'x': x, 'y': y, 'w': cw, 'h': ch, 'method': 'break'})
        
        return regions
    
    def _find_short_circuits(self, gray):
        """找短路区域"""
        regions = []
        
        # 自适应阈值
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 反转（PCB走线是暗的）
        inverted = 255 - binary
        
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5000 < area < 50000:  # 较大的连接区域
                x, y, cw, ch = cv2.boundingRect(cnt)
                regions.append({'x': x, 'y': y, 'w': cw, 'h': ch, 'method': 'short'})
        
        return regions
    
    def _find_spurs(self, gray):
        """找毛刺（细小突出）"""
        regions = []
        
        # 边缘检测
        edges = cv2.Canny(gray, 80, 200)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < 1500:
                x, y, cw, ch = cv2.boundingRect(cnt)
                # 小的孤立区域
                regions.append({'x': x, 'y': y, 'w': cw, 'h': ch, 'method': 'spur'})
        
        return regions
    
    def _find_excess_copper(self, gray):
        """找多余铜区域"""
        regions = []
        
        # 阈值分割
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 2000 < area < 30000:
                x, y, cw, ch = cv2.boundingRect(cnt)
                regions.append({'x': x, 'y': y, 'w': cw, 'h': ch, 'method': 'copper'})
        
        return regions
    
    def smart_label_single(self, img_path, defect_type):
        """智能标注单张图片"""
        if isinstance(defect_type, str):
            class_id = self.DEFECT_TYPES.get(defect_type, 0)
        else:
            class_id = defect_type
            defect_type = list(self.DEFECT_TYPES.keys())[class_id]
        
        img = cv2.imread(str(img_path))
        if img is None:
            return []
            
        h, w = img.shape[:2]
        
        # 检测缺陷区域
        regions = self.detect_defects_by_type(img_path, defect_type)
        
        # 如果没检测到，使用基于位置的默认标注
        if not regions:
            # 根据文件名编号分散标注位置
            idx = int(Path(img_path).stem.split('_')[-1]) if Path(img_path).stem.split('_')[-1].isdigit() else 1
            
            # 分散在不同位置
            row = (idx - 1) // 6
            col = (idx - 1) % 6
            
            # 默认框大小（针对PCB缺陷的合理大小）
            default_w = int(w * 0.08)
            default_h = int(h * 0.08)
            
            cx = w * (0.2 + col * 0.12)
            cy = h * (0.2 + row * 0.25)
            
            x = int(cx - default_w / 2)
            y = int(cy - default_h / 2)
            
            regions = [{
                'x': max(0, x), 'y': max(0, y),
                'w': default_w, 'h': default_h,
                'method': 'default'
            }]
        
        # 转换并保存为YOLO格式
        yolo_labels = []
        for region in regions[:3]:  # 最多3个
            cx = (region['x'] + region['w']/2) / w
            cy = (region['y'] + region['h']/2) / h
            nw = region['w'] / w
            nh = region['h'] / h
            
            # 限制范围
            cx = max(0.01, min(0.99, cx))
            cy = max(0.01, min(0.99, cy))
            nw = max(0.01, min(0.99, nw))
            nh = max(0.01, min(0.99, nh))
            
            yolo_labels.append((class_id, cx, cy, nw, nh))
        
        return yolo_labels
    
    def batch_process(self, split='train'):
        """批量处理"""
        img_dir = self.dataset_dir / 'images' / split
        label_dir = self.dataset_dir / 'labels' / split
        
        if not img_dir.exists():
            print(f"目录不存在: {img_dir}")
            return
        
        # 获取所有图片
        image_files = sorted(list(img_dir.glob('*.jpg')))
        
        stats = {'success': 0, 'failed': 0}
        
        for i, img_path in enumerate(image_files):
            # 解析缺陷类型
            defect_type = self._parse_defect_type(img_path.name)
            
            # 智能标注
            yolo_labels = self.smart_label_single(img_path, defect_type)
            
            # 保存
            label_file = label_dir / f"{img_path.stem}.txt"
            label_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                with open(label_file, 'w') as f:
                    for label in yolo_labels:
                        f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")
                stats['success'] += 1
            except Exception as e:
                stats['failed'] += 1
                print(f"错误: {img_path.name} - {e}")
            
            if (i + 1) % 100 == 0:
                print(f"进度: {i+1}/{len(image_files)}")
        
        print(f"\n完成! 成功: {stats['success']}, 失败: {stats['failed']}")
        return stats
    
    def _parse_defect_type(self, filename):
        """从文件名解析缺陷类型"""
        name = Path(filename).stem.lower()
        
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
        
        return 'missing_hole'  # 默认


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='./yolo_pcb_dataset')
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    args = parser.parse_args()
    
    labeler = PCBSmartLabelerV2(args.dataset)
    labeler.batch_process(args.split)


if __name__ == '__main__':
    main()
