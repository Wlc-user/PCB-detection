"""
智能标注工具 - 领域驱动 + 近似表征
====================================
核心理念:
1. 快速标注 - 点/线/面一键切换
2. 领域专用 - PCB/医学/遥感/工业 不同模板
3. 近似优先 - 不追求精确，后续算法优化
4. AI辅助 - 自动预标注 + 批量处理
"""
import cv2
import numpy as np
import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import threading
import time

# ============== 领域配置 ==============

DOMAIN_PROFILES = {
    "PCB检测": {
        "icon": "🔬",
        "shortcut": "p",
        "shapes": ["rect", "polygon"],  # 优先矩形
        "categories": [
            {"id": 0, "name": "defect", "cn": "缺陷", "color": "#FF0000", "priority": 1},
            {"id": 1, "name": "component", "cn": "元件", "color": "#00FF00", "priority": 2},
            {"id": 2, "name": "trace", "cn": "走线", "color": "#FFFF00", "priority": 3},
            {"id": 3, "name": "pad", "cn": "焊盘", "color": "#00FFFF", "priority": 4},
        ],
        "quick_annotation": True,  # 近似标注模式
        "snap_to_edge": True,     # 边缘吸附
        "auto_refine": True,      # 自动优化
    },
    
    "医学影像": {
        "icon": "🏥",
        "shortcut": "m",
        "shapes": ["brush", "polygon", "circle"],
        "categories": [
            {"id": 0, "name": "tumor", "cn": "肿瘤", "color": "#FF1493", "priority": 1},
            {"id": 1, "name": "lesion", "cn": "病灶", "color": "#FF6B6B", "priority": 2},
            {"id": 2, "name": "organ", "cn": "器官", "color": "#4ECDC4", "priority": 3},
        ],
        "quick_annotation": True,
        "brush_mode": True,  # 画笔模式
    },
    
    "遥感影像": {
        "icon": "🛰️",
        "shortcut": "r",
        "shapes": ["polygon", "rect"],
        "categories": [
            {"id": 0, "name": "building", "cn": "建筑", "color": "#FF6B6B", "priority": 1},
            {"id": 1, "name": "road", "cn": "道路", "color": "#FFFF00", "priority": 2},
            {"id": 2, "name": "vegetation", "cn": "植被", "color": "#00FF00", "priority": 3},
            {"id": 3, "name": "water", "cn": "水体", "color": "#0066FF", "priority": 4},
        ],
        "quick_annotation": True,
        "auto_merge": True,  # 自动合并相邻区域
    },
    
    "通用物体": {
        "icon": "📦",
        "shortcut": "g",
        "shapes": ["rect", "polygon", "line"],
        "categories": [
            {"id": 0, "name": "object", "cn": "物体", "color": "#00FF00", "priority": 1},
            {"id": 1, "name": "person", "cn": "人", "color": "#FF0000", "priority": 2},
            {"id": 2, "name": "vehicle", "cn": "车辆", "color": "#0000FF", "priority": 3},
        ],
        "quick_annotation": False,
    },
    
    "工业缺陷": {
        "icon": "⚙️",
        "shortcut": "i",
        "shapes": ["rect", "polygon", "line", "point"],
        "categories": [
            {"id": 0, "name": "scratch", "cn": "划痕", "color": "#FF0000", "priority": 1},
            {"id": 1, "name": "dent", "cn": "凹坑", "color": "#FF6600", "priority": 2},
            {"id": 2, "name": "crack", "cn": "裂纹", "color": "#990000", "priority": 3},
            {"id": 3, "name": "stain", "cn": "污渍", "color": "#999999", "priority": 4},
        ],
        "quick_annotation": True,
        "edge_enhance": True,  # 边缘增强辅助
    },
}


class SmartLabelTool:
    """智能标注工具"""
    
    def __init__(self):
        # 当前领域
        self.current_domain = "PCB检测"
        self.domain_config = DOMAIN_PROFILES[self.current_domain]
        
        # 类别
        self.categories = self.domain_config["categories"]
        self.current_category = 0
        
        # 标注模式
        self.shape_mode = "rect"  # rect, polygon, line, point, brush
        self.quick_mode = self.domain_config.get("quick_annotation", False)
        
        # 数据
        self.current_image = None
        self.image_path = None
        self.image_h, self.image_w = 0, 0
        self.annotations: List[dict] = []
        
        # 绘制状态
        self.drawing = False
        self.start_point = (0, 0)
        self.current_shape = []
        self.current_box = None
        self.selected_idx = -1
        
        # 显示
        self.display_scale = 1.0
        self.zoom_level = 1.0
        
        # AI预标注
        self.ai_enabled = False
        self.ai_model = None
        
        self._create_ui()
    
    def _create_ui(self):
        """创建UI"""
        self.root = tk.Tk()
        self.root.title(f"智能标注工具 - {self.current_domain} {self.domain_config['icon']}")
        self.root.geometry("1500x950")
        self.root.configure(bg="#1a1a2e")
        
        # 顶部工具栏
        toolbar = tk.Frame(self.root, bg="#16213e", height=60)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        toolbar.pack_propagate(False)
        
        # 领域选择
        domain_frame = tk.Frame(toolbar, bg="#16213e")
        domain_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(domain_frame, text="领域:", font=("Arial", 11), 
                bg="#16213e", fg="white").pack(side=tk.LEFT, pady=15)
        
        domain_names = [f"{v['icon']} {k}" for k, v in DOMAIN_PROFILES.items()]
        self.domain_var = tk.StringVar(value=domain_names[0])
        domain_combo = ttk.Combobox(domain_frame, textvariable=self.domain_var,
                                   values=domain_names, width=15, state="readonly")
        domain_combo.pack(side=tk.LEFT, padx=5)
        domain_combo.bind('<<ComboboxSelected>>', self._on_domain_change)
        
        # 形状选择
        shape_frame = tk.Frame(toolbar, bg="#16213e")
        shape_frame.pack(side=tk.LEFT, padx=20)
        
        shapes = [
            ("矩形", "rect", "r"),
            ("多边形", "polygon", "p"),
            ("线段", "line", "l"),
            ("点", "point", "o"),
            ("画笔", "brush", "b"),
        ]
        
        for text, mode, key in shapes:
            btn = tk.Button(shape_frame, text=f"{text}({key})", 
                          font=("Arial", 9), width=6,
                          bg="#0f3460", fg="white", relief=tk.RAISED,
                          command=lambda m=mode: self._set_shape_mode(m))
            btn.pack(side=tk.LEFT, padx=2)
        
        # 快捷操作
        quick_frame = tk.Frame(toolbar, bg="#16213e")
        quick_frame.pack(side=tk.RIGHT, padx=10)
        
        quick_btns = [
            ("AI预标注", self._run_ai_annotation),
            ("批量标注", self._batch_annotate),
            ("自动优化", self._auto_refine),
        ]
        
        for text, cmd in quick_btns:
            tk.Button(quick_frame, text=text, font=("Arial", 9),
                     bg="#e94560", fg="white", command=cmd).pack(side=tk.LEFT, padx=2)
        
        # 主区域
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg="#1a1a2e")
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        # 左侧面板
        left_panel = tk.Frame(main_paned, width=260, bg="#16213e")
        main_paned.add(left_panel)
        
        # 类别列表
        cat_title = tk.Label(left_panel, text="标注类别", font=("Arial", 12, "bold"),
                            bg="#16213e", fg="#e94560")
        cat_title.pack(pady=(15, 5))
        
        self.cat_buttons = []
        for i, cat in enumerate(self.categories):
            color = cat['color']
            btn = tk.Button(left_panel, 
                          text=f"{i+1} {cat['cn']} ({cat['name']})",
                          font=("Arial", 10), anchor=tk.W,
                          bg=color, fg="black" if color != "#FFFF00" else "black",
                          activebackground=color, relief=tk.RAISED,
                          command=lambda idx=i: self._select_category(idx))
            btn.pack(fill=tk.X, padx=10, pady=2)
            self.cat_buttons.append(btn)
        
        self.cat_buttons[0].config(relief=tk.SUNKEN)
        
        # 标注预览
        preview_frame = tk.LabelFrame(left_panel, text="标注预览", 
                                     font=("Arial", 10), bg="#16213e", fg="white")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.preview_listbox = tk.Listbox(preview_frame, bg="#0f3460", fg="white",
                                         font=("Arial", 9))
        self.preview_listbox.pack(fill=tk.BOTH, expand=True)
        
        # 文件列表
        file_frame = tk.LabelFrame(left_panel, text="图片列表",
                                   font=("Arial", 10), bg="#16213e", fg="white")
        file_frame.pack(fill=tk.BOTH, side=tk.BOTTOM, padx=10, pady=10, ipady=100)
        
        file_scroll = tk.Scrollbar(file_frame)
        file_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.file_listbox = tk.Listbox(file_frame, yscrollcommand=file_scroll.set,
                                      bg="#0f3460", fg="white", font=("Arial", 9))
        self.file_listbox.pack(fill=tk.BOTH, expand=True)
        file_scroll.config(command=self.file_listbox.yview)
        self.file_listbox.bind('<Double-Button-1>', self._on_file_select)
        
        # 画布区域
        canvas_frame = tk.Frame(main_paned, bg="#1a1a2e")
        main_paned.add(canvas_frame)
        
        # 带滚动条的画布
        h_scroll = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        v_scroll = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        
        self.canvas = tk.Canvas(canvas_frame, bg="#2d2d44",
                               xscrollcommand=h_scroll.set,
                               yscrollcommand=v_scroll.set,
                               cursor="crosshair")
        
        h_scroll.config(command=self.canvas.xview)
        v_scroll.config(command=self.canvas.yview)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 鼠标事件
        self.canvas.bind('<Button-1>', self._on_mouse_down)
        self.canvas.bind('<B1-Motion>', self._on_mouse_move)
        self.canvas.bind('<ButtonRelease-1>', self._on_mouse_up)
        self.canvas.bind('<Double-Button-1>', self._on_double_click)  # 完成多边形
        self.canvas.bind('<MouseWheel>', self._on_wheel)
        self.canvas.bind('<Button-3>', self._on_right_click)
        
        # 键盘事件
        self.root.bind('<Key>', self._on_key)
        self.root.bind('<Control-s>', lambda e: self._save())
        self.root.bind('<Control-z>', lambda e: self._undo())
        self.root.bind('<Delete>', lambda e: self._delete())
        self.root.bind('<Control-a>', lambda e: self._open_folder())
        self.root.bind('<Control-n>', lambda e: self._next_image())
        self.root.bind('<Control-p>', lambda e: self._prev_image())
        
        # 底部状态栏
        status_bar = tk.Frame(self.root, bg="#16213e", height=30)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar(value="就绪 | 领域:PCB检测 | 快速标注:ON")
        tk.Label(status_bar, textvariable=self.status_var,
                font=("Arial", 9), bg="#16213e", fg="#4ecca3").pack(side=tk.LEFT, padx=10)
        
        self.info_var = tk.StringVar(value="")
        tk.Label(status_bar, textvariable=self.info_var,
                font=("Arial", 9), bg="#16213e", fg="white").pack(side=tk.RIGHT, padx=10)
        
        # 数据
        self.folder_path = None
        self.image_files = []
        self.current_idx = -1
        
        # 预加载缩略图
        self.thumbnails = {}
    
    def _on_domain_change(self, event):
        """切换领域"""
        domain_name = self.domain_var.get().split(" ", 1)[1]
        if domain_name in DOMAIN_PROFILES:
            self.current_domain = domain_name
            self.domain_config = DOMAIN_PROFILES[domain_name]
            self.categories = self.domain_config["categories"]
            self.shape_mode = self.domain_config["shapes"][0]
            self.quick_mode = self.domain_config.get("quick_annotation", False)
            
            self.root.title(f"智能标注工具 - {self.current_domain} {self.domain_config['icon']}")
            
            # 更新类别按钮
            for btn in self.cat_buttons:
                btn.destroy()
            self.cat_buttons = []
            
            for i, cat in enumerate(self.categories):
                btn = tk.Button(self.root.nametowidget(self.root.children['!panedwindow'].children['!frame'].children['!frame'].children['!frame']).winfo_children()[1] if hasattr(self, 'cat_buttons') else None,
                              text=f"{i+1} {cat['cn']}", font=("Arial", 10),
                              bg=cat['color'], fg="black", command=lambda idx=i: self._select_category(idx))
            
            # 简化：直接重建界面
            self._update_category_buttons()
            
            self._update_status(f"已切换到 {domain_name} 领域")
    
    def _update_category_buttons(self):
        """更新类别按钮"""
        for btn in self.cat_buttons:
            btn.destroy()
        self.cat_buttons = []
        
        left_panel = self.root.winfo_children()[1].winfo_children()[0]
        
        for i, cat in enumerate(self.categories):
            color = cat['color']
            btn = tk.Button(left_panel, 
                          text=f"{i+1} {cat['cn']} ({cat['name']})",
                          font=("Arial", 10), anchor=tk.W,
                          bg=color, fg="black",
                          activebackground=color, relief=tk.RAISED,
                          command=lambda idx=i: self._select_category(idx))
            btn.pack(fill=tk.X, padx=10, pady=2)
            self.cat_buttons.append(btn)
        
        self.cat_buttons[self.current_category].config(relief=tk.SUNKEN)
    
    def _set_shape_mode(self, mode: str):
        """设置形状模式"""
        self.shape_mode = mode
        self.current_shape = []
        self._update_status(f"当前形状: {mode}")
    
    def _select_category(self, idx: int):
        """选择类别"""
        self.cat_buttons[self.current_category].config(relief=tk.RAISED)
        self.current_category = idx
        self.cat_buttons[idx].config(relief=tk.SUNKEN)
        cat = self.categories[idx]
        self._update_status(f"已选择: {cat['cn']} ({cat['name']})")
    
    def _on_key(self, event):
        """键盘事件"""
        key = event.char
        
        # 数字键选择类别
        if key.isdigit():
            idx = int(key) - 1
            if idx < len(self.categories):
                self._select_category(idx)
        
        # 形状快捷键
        elif key in ['r', 'R']:
            self._set_shape_mode('rect')
        elif key in ['p', 'P']:
            self._set_shape_mode('polygon')
        elif key in ['l', 'L']:
            self._set_shape_mode('line')
        elif key in ['o', 'O']:
            self._set_shape_mode('point')
        elif key in ['b', 'B']:
            self._set_shape_mode('brush')
        
        # 快速标注开关
        elif key == 'q':
            self.quick_mode = not self.quick_mode
            self._update_status(f"快速标注: {'ON' if self.quick_mode else 'OFF'}")
        
        # 空格确认
        elif event.keysym == 'space':
            self._confirm_shape()
    
    def _open_folder(self):
        """打开文件夹"""
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path = folder
            self.image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                self.image_files.extend(Path(folder).glob(ext))
            
            self.file_listbox.delete(0, tk.END)
            for f in self.image_files:
                self.file_listbox.insert(tk.END, f.name)
            
            if self.image_files:
                self._load_image(str(self.image_files[0]))
                self.current_idx = 0
                self.file_listbox.selection_set(0)
            
            self._update_status(f"已加载 {len(self.image_files)} 张图片")
    
    def _load_image(self, path: str):
        """加载图片"""
        self.image_path = path
        self.current_image = cv2.imread(path)
        if self.current_image is not None:
            self.image_h, self.image_w = self.current_image.shape[:2]
            self.annotations = []
            self.current_shape = []
            self._load_existing_annotations()
            self._auto_scale()
            self._redraw()
            
            self.info_var.set(f"{self.image_w}x{self.image_h} | {len(self.annotations)} 标注")
    
    def _load_existing_annotations(self):
        """加载已有标注"""
        json_path = Path(self.image_path).with_suffix('.json')
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
                self.annotations = data.get('annotations', [])
    
    def _auto_scale(self):
        """自动缩放适应窗口"""
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w > 100 and canvas_h > 100:
            scale_w = canvas_w / self.image_w
            scale_h = canvas_h / self.image_h
            self.display_scale = min(scale_w, scale_h, 1.5)
    
    def _on_file_select(self, event):
        """选择文件"""
        idx = self.file_listbox.curselection()
        if idx:
            self.current_idx = idx[0]
            self._load_image(str(self.image_files[idx[0]]))
    
    def _next_image(self):
        """下一张"""
        if self.image_files and self.current_idx < len(self.image_files) - 1:
            self.current_idx += 1
            self._load_image(str(self.image_files[self.current_idx]))
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(self.current_idx)
    
    def _prev_image(self):
        """上一张"""
        if self.image_files and self.current_idx > 0:
            self.current_idx -= 1
            self._load_image(str(self.image_files[self.current_idx]))
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(self.current_idx)
    
    def _on_mouse_down(self, event):
        """鼠标按下"""
        if self.current_image is None:
            return
        
        self.drawing = True
        self.start_point = (event.x, event.y)
        
        if self.shape_mode == 'rect':
            self.current_box = [event.x, event.y, event.x, event.y]
        elif self.shape_mode == 'polygon':
            self.current_shape.append((event.x, event.y))
            self._redraw()
        elif self.shape_mode == 'point':
            self._add_point(event.x, event.y)
            self.drawing = False
        elif self.shape_mode == 'brush':
            self.current_shape = [(event.x, event.y)]
    
    def _on_mouse_move(self, event):
        """鼠标移动"""
        if not self.drawing:
            return
        
        if self.shape_mode == 'rect':
            self.current_box[2] = event.x
            self.current_box[3] = event.y
            self._redraw()
        elif self.shape_mode == 'brush':
            self.current_shape.append((event.x, event.y))
            self._redraw()
    
    def _on_mouse_up(self, event):
        """鼠标释放"""
        if not self.drawing:
            return
        
        self.drawing = False
        
        if self.shape_mode == 'rect' and self.current_box:
            x1, y1, x2, y2 = self.current_box
            # 转换为图像坐标
            ix1, iy1 = int(x1 / self.display_scale), int(y1 / self.display_scale)
            ix2, iy2 = int(x2 / self.display_scale), int(y2 / self.display_scale)
            
            if abs(ix2 - ix1) > 3 and abs(iy2 - iy1) > 3:
                # 快速标注模式：自动调整边界
                if self.quick_mode:
                    ix1, iy1, ix2, iy2 = self._quick_refine(ix1, iy1, ix2, iy2)
                
                self._add_bbox(ix1, iy1, ix2, iy2)
            
            self.current_box = None
            self._redraw()
    
    def _on_double_click(self, event):
        """双击完成多边形"""
        if self.shape_mode == 'polygon' and len(self.current_shape) > 2:
            self._confirm_polygon()
    
    def _on_right_click(self, event):
        """右键"""
        if self.shape_mode == 'polygon' and len(self.current_shape) > 2:
            self._confirm_polygon()
        else:
            self.current_shape = []
            self._redraw()
    
    def _on_wheel(self, event):
        """滚轮缩放"""
        if self.current_image is None:
            return
        
        delta = event.delta / 1200
        self.display_scale = max(0.2, min(5.0, self.display_scale + delta))
        self._redraw()
    
    def _quick_refine(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int, int, int]:
        """快速优化边界（近似算法）"""
        # 确保 x1 < x2, y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # 边缘吸附（如果启用）
        if self.domain_config.get("snap_to_edge"):
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            # 简化的边缘检测
            edges = cv2.Canny(gray, 50, 150)
            
            # 找最近的边缘
            for i in range(min(x1, 50)):
                if y1 < self.image_h and edges[y1, min(x1 - i, self.image_w - 1)] > 0:
                    x1 = x1 - i
                    break
            
            for i in range(min(self.image_w - x2, 50)):
                if y2 < self.image_h and edges[y2, min(x2 + i, self.image_w - 1)] > 0:
                    x2 = x2 + i
                    break
        
        return x1, y1, x2, y2
    
    def _add_bbox(self, x1: int, y1: int, x2: int, y2: int):
        """添加矩形"""
        cat = self.categories[self.current_category]
        self.annotations.append({
            "type": "bbox",
            "category_id": cat["id"],
            "category": cat["name"],
            "bbox": [x1, y1, x2, y2],
            "color": cat["color"]
        })
        self._update_preview()
    
    def _add_point(self, x: int, y: int):
        """添加点"""
        cat = self.categories[self.current_category]
        ix, iy = int(x / self.display_scale), int(y / self.display_scale)
        self.annotations.append({
            "type": "point",
            "category_id": cat["id"],
            "category": cat["name"],
            "point": [ix, iy],
            "color": cat["color"]
        })
        self._update_preview()
        self._redraw()
    
    def _confirm_polygon(self):
        """确认多边形"""
        if len(self.current_shape) < 3:
            return
        
        cat = self.categories[self.current_category]
        
        # 转换到图像坐标
        points = [[int(p[0] / self.display_scale), int(p[1] / self.display_scale)] 
                  for p in self.current_shape]
        
        self.annotations.append({
            "type": "polygon",
            "category_id": cat["id"],
            "category": cat["name"],
            "points": points,
            "color": cat["color"]
        })
        
        self.current_shape = []
        self._update_preview()
        self._redraw()
    
    def _confirm_shape(self):
        """确认当前形状"""
        if self.shape_mode == 'polygon':
            self._confirm_polygon()
    
    def _update_preview(self):
        """更新预览列表"""
        self.preview_listbox.delete(0, tk.END)
        for i, ann in enumerate(self.annotations):
            cat_name = ann.get('category', 'unknown')
            ann_type = ann.get('type', 'unknown')
            self.preview_listbox.insert(tk.END, f"{i+1}. [{ann_type}] {cat_name}")
    
    def _undo(self):
        """撤销"""
        if self.annotations:
            self.annotations.pop()
            self._update_preview()
            self._redraw()
    
    def _delete(self):
        """删除"""
        if self.annotations:
            self.annotations.pop()
            self._update_preview()
            self._redraw()
    
    def _auto_refine(self):
        """自动优化"""
        if not self.current_image:
            return
        
        self._update_status("正在优化标注...")
        
        for ann in self.annotations:
            if ann.get('type') == 'bbox':
                x1, y1, x2, y2 = ann['bbox']
                x1, y1, x2, y2 = self._quick_refine(x1, y1, x2, y2)
                ann['bbox'] = [x1, y1, x2, y2]
        
        self._redraw()
        self._update_status("优化完成")
    
    def _run_ai_annotation(self):
        """AI预标注"""
        self._update_status("AI预标注功能开发中...")
        messagebox.showinfo("AI标注", "AI预标注功能需要YOLO模型支持，请先训练模型。")
    
    def _batch_annotate(self):
        """批量标注"""
        messagebox.showinfo("批量标注", "批量标注功能开发中。")
    
    def _save(self):
        """保存"""
        if not self.image_path or not self.annotations:
            return
        
        json_path = Path(self.image_path).with_suffix('.json')
        data = {
            "image": os.path.basename(self.image_path),
            "width": self.image_w,
            "height": self.image_h,
            "domain": self.current_domain,
            "annotations": self.annotations,
            "saved_at": datetime.now().isoformat()
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self._update_status(f"已保存: {json_path.name}")
    
    def _export_yolo(self):
        """导出YOLO"""
        folder = filedialog.askdirectory(title="选择导出目录")
        if not folder:
            return
        
        for img_path in self.image_files:
            json_path = img_path.with_suffix('.json')
            if not json_path.exists():
                continue
            
            with open(json_path) as f:
                data = json.load(f)
            
            label_path = Path(folder) / (img_path.stem + ".txt")
            with open(label_path, 'w') as f:
                for ann in data.get('annotations', []):
                    if ann.get('type') == 'bbox':
                        x1, y1, x2, y2 = ann['bbox']
                        w, h = data['width'], data['height']
                        
                        cx = (x1 + x2) / 2 / w
                        cy = (y1 + y2) / 2 / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        
                        f.write(f"{ann['category_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        
        self._update_status(f"已导出到: {folder}")
    
    def _export_coco(self):
        """导出COCO"""
        folder = filedialog.askdirectory(title="选择导出目录")
        if not folder:
            return
        
        coco = {
            "images": [],
            "annotations": [],
            "categories": [{"id": c["id"], "name": c["name"]} for c in self.categories]
        }
        
        ann_id = 1
        for img_path in self.image_files:
            json_path = img_path.with_suffix('.json')
            if not json_path.exists():
                continue
            
            with open(json_path) as f:
                data = json.load(f)
            
            img_idx = len(coco["images"]) + 1
            coco["images"].append({
                "id": img_idx,
                "file_name": img_path.name,
                "width": data["width"],
                "height": data["height"]
            })
            
            for ann in data.get('annotations', []):
                if ann.get('type') == 'bbox':
                    x1, y1, x2, y2 = ann['bbox']
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_idx,
                        "category_id": ann["category_id"],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": 0
                    })
                    ann_id += 1
        
        output_path = Path(folder) / "annotations.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco, f, indent=2)
        
        self._update_status(f"已导出COCO到: {output_path}")
    
    def _update_status(self, text: str):
        """更新状态"""
        self.status_var.set(f"{text} | 领域:{self.current_domain} | 快速标注:{'ON' if self.quick_mode else 'OFF'}")
    
    def _redraw(self):
        """重绘画布"""
        if self.current_image is None:
            self.canvas.delete('all')
            return
        
        display = self.current_image.copy()
        h, w = display.shape[:2]
        
        # 绘制已有标注
        for ann in self.annotations:
            color_str = ann.get('color', '#00FF00')
            color = self._hex_to_bgr(color_str)
            
            if ann.get('type') == 'bbox':
                x1, y1, x2, y2 = ann['bbox']
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                
                label = ann.get('category', '')
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(display, (x1, y1-lh-4), (x1+lw+4, y1), color, -1)
                cv2.putText(display, label, (x1+2, y1-2), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            elif ann.get('type') == 'polygon':
                points = np.array(ann.get('points', []), np.int32)
                cv2.polylines(display, [points], True, color, 2)
                cv2.fillPoly(display, [points], color=[c//3 for c in color])
            
            elif ann.get('type') == 'point':
                px, py = ann.get('point', [0, 0])
                cv2.circle(display, (px, py), 5, color, -1)
        
        # 绘制当前正在绘制的形状
        if self.current_box:
            cv2.rectangle(display, (self.current_box[0], self.current_box[1]),
                        (self.current_box[2], self.current_box[3]), (0, 255, 255), 2)
        
        if self.current_shape and len(self.current_shape) > 1:
            pts = np.array(self.current_shape, np.int32)
            cv2.polylines(display, [pts], False, (255, 0, 255), 2)
            for pt in self.current_shape:
                cv2.circle(display, pt, 4, (255, 0, 255), -1)
        
        # 缩放
        new_w = int(w * self.display_scale)
        new_h = int(h * self.display_scale)
        resized = cv2.resize(display, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 显示
        from PIL import Image, ImageTk
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=(0, 0, new_w, new_h))
        
        self.info_var.set(f"{self.image_w}x{self.image_h} | {len(self.annotations)} 标注 | 缩放:{self.display_scale:.1f}x")
    
    def _hex_to_bgr(self, hex_color: str) -> Tuple[int, int, int]:
        """HEX转BGR"""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (b, g, r)
    
    def run(self):
        """运行"""
        self.root.mainloop()


def main():
    tool = SmartLabelTool()
    tool.run()


if __name__ == "__main__":
    main()
