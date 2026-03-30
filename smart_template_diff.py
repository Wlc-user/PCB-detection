"""
PCB缺陷检测 - 模板差分法
如果有无缺陷的模板图片，可以用这个方法生成准确的标注
"""
import cv2
import numpy as np
from pathlib import Path

class TemplateDiffDetector:
    """基于模板差分的缺陷检测"""
    
    def __init__(self, template_path=None):
        self.template = None
        if template_path and Path(template_path).exists():
            self.template = cv2.imread(str(template_path))
            print(f"已加载模板: {template_path}")
    
    def detect_by_difference(self, img_path, threshold=30, min_area=50):
        """通过与模板对比检测缺陷"""
        if self.template is None:
            print("警告: 未设置模板，使用默认方法")
            return []
        
        img = cv2.imread(str(img_path))
        if img is None:
            return []
        
        # 确保尺寸一致
        if img.shape != self.template.shape:
            img = cv2.resize(img, (self.template.shape[1], self.template.shape[0]))
        
        # 灰度化
        gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        
        # 差分
        diff = cv2.absdiff(gray1, gray2)
        
        # 二值化
        _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # 形态学操作
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        h, w = img.shape[:2]
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, cw, ch = cv2.boundingRect(cnt)
                # 扩大一点
                padding = 5
                x = max(0, x - padding)
                y = max(0, y - padding)
                cw = min(cw + padding*2, w - x)
                ch = min(ch + padding*2, h - y)
                regions.append((x, y, cw, ch))
        
        return regions
    
    def convert_to_yolo(self, bbox, img_width, img_height):
        """转换为YOLO格式"""
        x, y, w, h = bbox
        cx = (x + w/2) / img_width
        cy = (y + h/2) / img_height
        nw = w / img_width
        nh = h / img_height
        return cx, cy, nw, nh


def demo():
    """演示模板差分方法"""
    detector = TemplateDiffDetector()
    
    # 示例用法
    print("="*50)
    print("PCB缺陷检测 - 模板差分法")
    print("="*50)
    print("""
使用方法:
1. 准备一张无缺陷的标准PCB模板图片 (template.jpg)
2. 将有缺陷的图片放在 test/ 目录
3. 运行检测:

from smart_template_diff import TemplateDiffDetector

detector = TemplateDiffDetector('template.jpg')
regions = detector.detect_by_difference('defect_image.jpg')

# 转换为YOLO格式
for region in regions:
    cx, cy, nw, nh = detector.convert_to_yolo(region, 3034, 1586)
    print(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
""")
    print("="*50)


if __name__ == '__main__':
    demo()
