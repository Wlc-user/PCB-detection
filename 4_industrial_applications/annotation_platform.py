"""
专业标注工具 v4.0 - AI智能检测版
- 集成YOLO目标检测
- 自动识别重复模式
- 支持自定义模型
"""

import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import json
from pathlib import Path
import threading
from collections import defaultdict

# 尝试导入深度学习库
try:
    import torch
    import torchvision
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("警告: PyTorch未安装，使用传统检测方法")

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("警告: Ultralytics未安装，YOLO功能不可用")


# ============== AI检测器 ==============
class AIDetector:
    """AI目标检测器 - 支持多种模型"""
    
    def __init__(self):
        self.model = None
        self.model_type = None
        self.device = 'cuda' if HAS_TORCH and torch.cuda.is_available() else 'cpu'
        
    def load_yolo_model(self, model_path='yolov8n.pt'):
        """加载YOLO模型"""
        if not HAS_YOLO:
            print("请安装: pip install ultralytics")
            return False
        
        try:
            self.model = YOLO(model_path)
            self.model_type = 'yolo'
            print(f"YOLO模型加载成功，设备: {self.device}")
            return True
        except Exception as e:
            print(f"加载YOLO模型失败: {e}")
            return False
    
    def load_custom_model(self, model_path):
        """加载自定义PyTorch模型"""
        if not HAS_TORCH:
            print("请安装: pip install torch torchvision")
            return False
        
        try:
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            self.model_type = 'custom'
            return True
        except Exception as e:
            print(f"加载自定义模型失败: {e}")
            return False
    
    def detect(self, image, conf_threshold=0.5, iou_threshold=0.45):
        """执行目标检测"""
        if self.model is None:
            return []
        
        detections = []
        
        if self.model_type == 'yolo':
            # YOLO检测
            results = self.model(image, conf=conf_threshold, iou=iou_threshold)
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    for box, conf, cls in zip(boxes, confs, classes):
                        x1, y1, x2, y2 = map(int, box)
                        class_name = result.names[int(cls)]
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'class': class_name,
                            'class_id': int(cls)
                        })
        
        return detections


# ============== 传统检测器（备选） ==============
class TraditionalDetector:
    """传统图像处理方法"""
    
    @staticmethod
    def detect_by_contour(image, min_area=500):
        """基于轮廓检测"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            detections.append({
                'bbox': [x, y, x + w, y + h],
                'confidence': min(1.0, area / 1000),
                'class': 'object',
                'class_id': 0
            })
        
        return detections
    
    @staticmethod
    def detect_by_blob(image, min_area=500):
        """基于斑点检测"""
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 10
        params.maxThreshold = 200
        params.filterByArea = True
        params.minArea = min_area
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        
        detector = cv2.SimpleBlobDetector_create(params)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints = detector.detect(gray)
        
        detections = []
        for kp in keypoints:
            x = int(kp.pt[0] - kp.size / 2)
            y = int(kp.pt[1] - kp.size / 2)
            w = int(kp.size)
            h = int(kp.size)
            detections.append({
                'bbox': [x, y, x + w, y + h],
                'confidence': 0.7,
                'class': 'blob',
                'class_id': 0
            })
        
        return detections
    
    @staticmethod
    def detect_by_template(image, template, threshold=0.7):
        """基于模板匹配"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
        
        result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)
        
        detections = []
        h, w = template_gray.shape
        
        for pt in zip(*locations[::-1]):
            detections.append({
                'bbox': [pt[0], pt[1], pt[0] + w, pt[1] + h],
                'confidence': float(result[pt[1], pt[0]]),
                'class': 'template',
                'class_id': 0
            })
        
        # 非极大值抑制
        return TraditionalDetector.nms(detections, 0.3)
    
    @staticmethod
    def nms(detections, iou_threshold=0.3):
        """非极大值抑制"""
        if not detections:
            return []
        
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            to_remove = []
            for i, det in enumerate(detections):
                iou = TraditionalDetector.compute_iou(best['bbox'], det['bbox'])
                if iou > iou_threshold:
                    to_remove.append(i)
            
            for i in reversed(to_remove):
                detections.pop(i)
        
        return keep
    
    @staticmethod
    def compute_iou(box1, box2):
        """计算IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
        return iou


# ============== 画布组件 ==============
class Canvas(QLabel):
    """标注画布"""
    
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #555;")
        self.setMinimumSize(900, 650)
        
        self.original_image = None
        self.processed_image = None
        self.display_image = None
        self.image_path = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # 标注数据
        self.detections = []  # 检测结果
        
        # 当前绘制
        self.current_rect_start = None
        self.template_rect = None
        self.selecting_template = False
        
        # 交互状态
        self.selected_idx = -1
        self.dragging = False
        self.drag_start = None
        
        # AI检测器
        self.ai_detector = AIDetector()
        self.traditional_detector = TraditionalDetector()
        
        # 预处理参数
        self.filter_type = 'none'
        self.filter_kernel = 5
        
        self.undo_stack = []
        self.redo_stack = []
        
        self.setMouseTracking(True)
        
        # 尝试加载默认YOLO模型
        if HAS_YOLO:
            self.ai_detector.load_yolo_model('yolov8n.pt')
    
    def load_image(self, path):
        self.image_path = path
        img = cv2.imread(path)
        if img is None:
            return False
        self.original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.processed_image = self.original_image.copy()
        self.detections = []
        self.apply_preprocessing()
        self.fit_to_window()
        self.update_display()
        return True
    
    def apply_preprocessing(self):
        if self.original_image is None:
            return
        
        img = self.original_image.copy()
        
        if self.filter_type == 'gaussian':
            ks = self.filter_kernel if self.filter_kernel % 2 == 1 else self.filter_kernel + 1
            img = cv2.GaussianBlur(img, (ks, ks), 0)
        elif self.filter_type == 'median':
            ks = self.filter_kernel if self.filter_kernel % 2 == 1 else self.filter_kernel + 1
            img = cv2.medianBlur(img, ks)
        
        self.processed_image = img
        self.fit_to_window()
        self.update_display()
    
    def ai_detect(self, method='yolo', conf=0.5):
        """AI检测"""
        if self.original_image is None:
            return []
        
        if method == 'yolo' and self.ai_detector.model is not None:
            detections = self.ai_detector.detect(self.original_image, conf)
        elif method == 'contour':
            detections = self.traditional_detector.detect_by_contour(self.original_image)
        elif method == 'blob':
            detections = self.traditional_detector.detect_by_blob(self.original_image)
        else:
            return []
        
        self.save_state()
        for det in detections:
            self.detections.append(det)
        
        self.update_display()
        return detections
    
    def detect_by_template(self, template_rect):
        """模板匹配检测"""
        x1, y1, x2, y2 = template_rect
        template = self.original_image[y1:y2, x1:x2]
        
        detections = self.traditional_detector.detect_by_template(
            self.original_image, template, threshold=0.7
        )
        
        self.save_state()
        for det in detections:
            self.detections.append(det)
        
        self.update_display()
        return detections
    
    def save_state(self):
        state = [d.copy() for d in self.detections]
        self.undo_stack.append(state)
        self.redo_stack.clear()
    
    def undo(self):
        if self.undo_stack:
            self.redo_stack.append([d.copy() for d in self.detections])
            self.detections = self.undo_stack.pop()
            self.update_display()
    
    def redo(self):
        if self.redo_stack:
            self.undo_stack.append([d.copy() for d in self.detections])
            self.detections = self.redo_stack.pop()
            self.update_display()
    
    def clear_all(self):
        self.save_state()
        self.detections = []
        self.update_display()
    
    def delete_selected(self):
        if self.selected_idx >= 0 and self.selected_idx < len(self.detections):
            self.save_state()
            del self.detections[self.selected_idx]
            self.selected_idx = -1
            self.update_display()
    
    def fit_to_window(self):
        if self.processed_image is None:
            return
        h, w = self.processed_image.shape[:2]
        win_w = self.width()
        win_h = self.height()
        if win_w <= 0 or win_h <= 0:
            return
        self.scale = min(win_w / w, win_h / h, 1.0)
        new_w = max(1, int(w * self.scale))
        new_h = max(1, int(h * self.scale))
        self.display_image = cv2.resize(self.processed_image, (new_w, new_h))
        self.offset_x = (win_w - new_w) // 2
        self.offset_y = (win_h - new_h) // 2
    
    def resizeEvent(self, event):
        if self.processed_image is not None:
            self.fit_to_window()
            self.update_display()
    
    def canvas_to_image(self, x, y):
        if self.processed_image is None:
            return None
        
        rel_x = x - self.offset_x
        rel_y = y - self.offset_y
        
        if rel_x < 0 or rel_y < 0:
            return None
        
        img_x = int(rel_x / self.scale)
        img_y = int(rel_y / self.scale)
        
        h, w = self.processed_image.shape[:2]
        if 0 <= img_x < w and 0 <= img_y < h:
            return [img_x, img_y]
        return None
    
    def image_to_canvas(self, x, y):
        return (int(x * self.scale + self.offset_x), int(y * self.scale + self.offset_y))
    
    def select_template(self):
        self.selecting_template = True
        self.template_rect = None
    
    def hit_test(self, canvas_pos):
        for i, det in enumerate(self.detections):
            x1, y1, x2, y2 = det['bbox']
            if x1 <= canvas_pos[0] <= x2 and y1 <= canvas_pos[1] <= y2:
                return i
        return -1
    
    def mousePressEvent(self, event):
        if self.processed_image is None:
            return
        
        canvas_pos = (event.pos().x(), event.pos().y())
        img_pos = self.canvas_to_image(canvas_pos[0], canvas_pos[1])
        
        if img_pos is None:
            return
        
        # 右键删除
        if event.button() == Qt.RightButton:
            hit_idx = self.hit_test(canvas_pos)
            if hit_idx >= 0:
                self.save_state()
                del self.detections[hit_idx]
                if self.selected_idx == hit_idx:
                    self.selected_idx = -1
                self.update_display()
            return
        
        # 左键
        if event.button() == Qt.LeftButton:
            # 模板选择模式
            if self.selecting_template:
                if self.template_rect is None:
                    self.template_rect = [img_pos[0], img_pos[1], img_pos[0], img_pos[1]]
                else:
                    self.template_rect[2] = img_pos[0]
                    self.template_rect[3] = img_pos[1]
                    self.detect_by_template(self.template_rect)
                    self.selecting_template = False
                    self.template_rect = None
                self.update_display()
                return
            
            # 检查是否点到已有检测框
            hit_idx = self.hit_test(canvas_pos)
            if hit_idx >= 0:
                self.selected_idx = hit_idx
                self.dragging = True
                self.drag_start = img_pos
                self.update_display()
                return
            
            # 开始新矩形
            if self.current_rect_start is None:
                self.current_rect_start = img_pos
            else:
                self.save_state()
                x1, y1 = self.current_rect_start
                x2, y2 = img_pos
                self.detections.append({
                    'bbox': [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
                    'confidence': 1.0,
                    'class': 'manual',
                    'class_id': -1
                })
                self.current_rect_start = None
                self.update_display()
            self.update_display()
    
    def mouseMoveEvent(self, event):
        if self.dragging and self.selected_idx >= 0:
            img_pos = self.canvas_to_image(event.pos().x(), event.pos().y())
            if img_pos and self.drag_start:
                dx = img_pos[0] - self.drag_start[0]
                dy = img_pos[1] - self.drag_start[1]
                bbox = self.detections[self.selected_idx]['bbox']
                self.detections[self.selected_idx]['bbox'] = [
                    bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy
                ]
                self.drag_start = img_pos
                self.update_display()
        elif self.current_rect_start:
            img_pos = self.canvas_to_image(event.pos().x(), event.pos().y())
            if img_pos:
                self.update_display()
        elif self.selecting_template and self.template_rect:
            img_pos = self.canvas_to_image(event.pos().x(), event.pos().y())
            if img_pos:
                self.template_rect[2] = img_pos[0]
                self.template_rect[3] = img_pos[1]
                self.update_display()
    
    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.drag_start = None
    
    def update_display(self):
        if self.display_image is None:
            return
        
        display = self.display_image.copy()
        
        # 绘制检测框
        for i, det in enumerate(self.detections):
            x1, y1, x2, y2 = det['bbox']
            p1 = self.image_to_canvas(x1, y1)
            p2 = self.image_to_canvas(x2, y2)
            
            # 颜色：选中的为黄色，手动标注为绿色，AI检测为蓝色
            if i == self.selected_idx:
                color = (0, 255, 255)  # 黄色
                thickness = 3
            elif det['class'] == 'manual':
                color = (0, 255, 0)    # 绿色
                thickness = 2
            else:
                color = (255, 100, 0)  # 蓝色
                thickness = 2
            
            cv2.rectangle(display, p1, p2, color, thickness)
            
            # 显示置信度和类别
            if det['confidence'] < 1.0:
                label = f"{det['class']}: {det['confidence']:.2f}"
            else:
                label = det['class']
            
            cv2.putText(display, label, (p1[0], p1[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 绘制当前矩形
        if self.current_rect_start:
            start = self.image_to_canvas(self.current_rect_start[0], self.current_rect_start[1])
            end = (event.pos().x(), event.pos().y()) if hasattr(self, 'event') else start
            cv2.rectangle(display, start, end, (255, 255, 0), 2)
        
        # 绘制模板选择矩形
        if self.selecting_template and self.template_rect:
            x1, y1, x2, y2 = self.template_rect
            p1 = self.image_to_canvas(x1, y1)
            p2 = self.image_to_canvas(x2, y2)
            cv2.rectangle(display, p1, p2, (255, 0, 255), 2)
            cv2.putText(display, "选择模板", (p1[0], p1[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # 显示
        h, w, ch = display.shape
        bytes_line = ch * w
        qt_img = QImage(display.data, w, h, bytes_line, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qt_img))


# ============== 主窗口 ==============
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI智能标注工具 v4.0")
        self.setGeometry(100, 100, 1400, 900)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout()
        central.setLayout(layout)
        
        # 左侧面板
        left_panel = self.create_left_panel()
        
        scroll = QScrollArea()
        scroll.setWidget(left_panel)
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(380)
        
        self.canvas = Canvas()
        
        layout.addWidget(scroll)
        layout.addWidget(self.canvas, 1)
        
        self.setup_shortcuts()
        self.statusBar().showMessage("AI标注工具 | 支持YOLO检测、轮廓检测、斑点检测")
    
    def create_left_panel(self):
        panel = QWidget()
        panel.setStyleSheet("""
            QWidget { background-color: #2d2d2d; color: white; }
            QGroupBox { color: white; border: 1px solid #555; border-radius: 5px;
                        margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background-color: #3c3c3c; border: 1px solid #555;
                          border-radius: 3px; padding: 8px; color: white; }
            QPushButton:hover { background-color: #4a4a4a; }
            QComboBox, QSpinBox, QDoubleSpinBox { background-color: #3c3c3c; 
                border: 1px solid #555; border-radius: 3px; padding: 5px; color: white;
                min-width: 100px; }
            QSlider::groove:horizontal { height: 4px; background: #555; }
            QSlider::handle:horizontal { background: #0d7377; width: 12px;
                                          border-radius: 6px; }
            QListWidget { background-color: #3c3c3c; border: 1px solid #555;
                          color: white; }
            QLabel { color: #cccccc; }
        """)
        
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # 文件操作
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout()
        self.btn_load = QPushButton("📂 加载图像")
        self.btn_load.clicked.connect(self.load_image)
        self.btn_save = QPushButton("💾 保存标注")
        self.btn_save.clicked.connect(self.save_annotations)
        file_layout.addWidget(self.btn_load)
        file_layout.addWidget(self.btn_save)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # 预处理
        pre_group = QGroupBox("图像预处理")
        pre_layout = QVBoxLayout()
        pre_layout.addWidget(QLabel("滤波:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["无", "高斯滤波", "中值滤波"])
        self.filter_combo.currentIndexChanged.connect(self.on_filter_changed)
        pre_layout.addWidget(self.filter_combo)
        
        self.kernel_slider = QSlider(Qt.Horizontal)
        self.kernel_slider.setRange(3, 15)
        self.kernel_slider.setValue(5)
        self.kernel_slider.valueChanged.connect(self.on_kernel_changed)
        pre_layout.addWidget(self.kernel_slider)
        pre_group.setLayout(pre_layout)
        layout.addWidget(pre_group)
        
        # AI检测
        ai_group = QGroupBox("AI智能检测")
        ai_layout = QVBoxLayout()
        
        ai_layout.addWidget(QLabel("检测方法:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["YOLO (深度学习)", "轮廓检测", "斑点检测"])
        ai_layout.addWidget(self.method_combo)
        
        ai_layout.addWidget(QLabel("置信度阈值:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(30, 95)
        self.conf_slider.setValue(50)
        self.conf_label = QLabel("0.50")
        self.conf_slider.valueChanged.connect(lambda v: self.conf_label.setText(f"{v/100:.2f}"))
        ai_layout.addWidget(self.conf_slider)
        ai_layout.addWidget(self.conf_label)
        
        self.btn_detect = QPushButton("🎯 开始AI检测")
        self.btn_detect.clicked.connect(self.ai_detect)
        ai_layout.addWidget(self.btn_detect)
        
        ai_group.setLayout(ai_layout)
        layout.addWidget(ai_group)
        
        # 模板匹配
        template_group = QGroupBox("模板匹配")
        template_layout = QVBoxLayout()
        
        self.btn_template = QPushButton("✂️ 框选模板并匹配")
        self.btn_template.clicked.connect(self.select_template)
        template_layout.addWidget(self.btn_template)
        
        template_group.setLayout(template_layout)
        layout.addWidget(template_group)
        
        # 手动标注
        manual_group = QGroupBox("手动标注")
        manual_layout = QVBoxLayout()
        
        self.btn_delete = QPushButton("🗑 删除选中")
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_clear = QPushButton("清除所有")
        self.btn_clear.clicked.connect(self.clear_all)
        
        manual_layout.addWidget(self.btn_delete)
        manual_layout.addWidget(self.btn_clear)
        manual_group.setLayout(manual_layout)
        layout.addWidget(manual_group)
        
        # 编辑
        edit_group = QGroupBox("编辑")
        edit_layout = QVBoxLayout()
        self.btn_undo = QPushButton("↩️ 撤销")
        self.btn_undo.clicked.connect(lambda: self.canvas.undo())
        self.btn_redo = QPushButton("↪️ 重做")
        self.btn_redo.clicked.connect(lambda: self.canvas.redo())
        edit_layout.addWidget(self.btn_undo)
        edit_layout.addWidget(self.btn_redo)
        edit_group.setLayout(edit_layout)
        layout.addWidget(edit_group)
        
        # 统计
        stats_group = QGroupBox("统计")
        stats_layout = QVBoxLayout()
        self.stats_label = QLabel("检测数量: 0")
        stats_layout.addWidget(self.stats_label)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # 标注列表
        list_group = QGroupBox("检测列表")
        list_layout = QVBoxLayout()
        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self.on_list_click)
        list_layout.addWidget(self.list_widget)
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)
        
        # 安装说明
        if not HAS_YOLO:
            info_group = QGroupBox("提示")
            info_layout = QVBoxLayout()
            info_layout.addWidget(QLabel("安装YOLO以获得更好效果:"))
            info_layout.addWidget(QLabel("pip install ultralytics"))
            info_group.setLayout(info_layout)
            layout.addWidget(info_group)
        
        layout.addStretch()
        return panel
    
    def setup_shortcuts(self):
        QShortcut(QKeySequence("Delete"), self, lambda: self.delete_selected())
        QShortcut(QKeySequence("Ctrl+Z"), self, lambda: self.canvas.undo())
        QShortcut(QKeySequence("Ctrl+Y"), self, lambda: self.canvas.redo())
    
    def on_filter_changed(self):
        filters = ["none", "gaussian", "median"]
        self.canvas.filter_type = filters[self.filter_combo.currentIndex()]
        self.canvas.apply_preprocessing()
    
    def on_kernel_changed(self):
        kernel = self.kernel_slider.value()
        if kernel % 2 == 0:
            kernel += 1
            self.kernel_slider.setValue(kernel)
        self.canvas.filter_kernel = kernel
        self.canvas.apply_preprocessing()
    
    def ai_detect(self):
        if self.canvas.original_image is None:
            QMessageBox.warning(self, "警告", "请先加载图像")
            return
        
        method_idx = self.method_combo.currentIndex()
        methods = ['yolo', 'contour', 'blob']
        method = methods[method_idx]
        conf = self.conf_slider.value() / 100
        
        self.canvas.save_state()
        detections = self.canvas.ai_detect(method, conf)
        
        self.update_list()
        self.update_stats()
        
        if detections:
            self.statusBar().showMessage(f"检测到 {len(detections)} 个目标")
        else:
            self.statusBar().showMessage("未检测到目标，请调整参数或更换方法")
    
    def select_template(self):
        if self.canvas.original_image is None:
            QMessageBox.warning(self, "警告", "请先加载图像")
            return
        
        self.canvas.select_template()
        self.statusBar().showMessage("请框选一个目标作为模板")
    
    def delete_selected(self):
        self.canvas.delete_selected()
        self.update_list()
        self.update_stats()
    
    def clear_all(self):
        if QMessageBox.question(self, "确认", "清除所有标注？") == QMessageBox.Yes:
            self.canvas.clear_all()
            self.update_list()
            self.update_stats()
    
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path and self.canvas.load_image(path):
            self.update_list()
            self.update_stats()
            self.setWindowTitle(f"AI标注工具 - {Path(path).name}")
            self.statusBar().showMessage(f"已加载: {Path(path).name}")
    
    def save_annotations(self):
        if not self.canvas.image_path:
            QMessageBox.warning(self, "警告", "请先加载图像")
            return
        
        default_path = Path(self.canvas.image_path).with_suffix('.json')
        path, _ = QFileDialog.getSaveFileName(self, "保存标注", str(default_path), "JSON files (*.json)")
        if path:
            data = {
                'image_path': self.canvas.image_path,
                'detections': self.canvas.detections
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            self.statusBar().showMessage(f"已保存: {path}")
    
    def update_list(self):
        self.list_widget.clear()
        for i, det in enumerate(self.canvas.detections):
            bbox = det['bbox']
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            self.list_widget.addItem(f"{i+1}. {det['class']} ({w}x{h}) conf:{det['confidence']:.2f}")
    
    def update_stats(self):
        self.stats_label.setText(f"检测数量: {len(self.canvas.detections)}")
    
    def on_list_click(self, item):
        idx = self.list_widget.row(item)
        self.canvas.selected_idx = idx
        self.canvas.update_display()


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()