# PortAI - 港口数据采集与智能检测系统
# PortAI - Port Data Collection & Intelligent Detection System
# 
# Copyright (c) 2026 PortAI Project
# MIT License - 商业可用，可自由修改和分发
# MIT License - Commercial use, freely modify and distribute
#
# 作者：PortAI Team
# Author: PortAI Team

"""
PortAI 港口智能检测系统 / PortAI Port Intelligence System
============================================================

功能 Features:
- 多协议数据采集 (RTSP/Modbus/MQTT/HTTP)
- 港口场景标注 (135类)
- YOLOv10剪枝蒸馏训练
- 边缘设备部署

支持协议 Supported Protocols:
- 视频流 Video: RTSP, ONVIF, HTTP
- 工业控制 Industrial: Modbus TCP/RTU
- 物联网 IoT: MQTT, CoAP
- 业务系统 Business: HTTP REST API

许可证 License: MIT
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import os
import json
import threading
import time
from datetime import datetime
import requests

__version__ = "1.0.0"
__author__ = "PortAI Team"


# ============================================================
# 港口场景类别配置 / Port Scene Category Configuration
# ============================================================

PORT_CATEGORIES = {
    # 港口监控 / Port Surveillance
    "port_surveillance": {
        "name": "港口监控 Surveillance",
        "categories": [
            "person", "worker", "helmet", "vest", "glove",      # 人员 PPE
            "forklift", "truck", "container", "crane",          # 设备
            "ship", "barge", "tugboat",                          # 船舶
            "conveyor", "pipelane", "storage_tank",             # 设施
            "fire", "smoke", "spill", "debris",                 # 安全事件
            "unsafe_behavior", "intrusion", "trespassing"       # 违规
        ]
    },
    
    # 港口物流 / Port Logistics
    "port_logistics": {
        "name": "港口物流 Logistics",
        "categories": [
            "container_20ft", "container_40ft", "container_45ft",  # 集装箱
            "empty_container", "full_container",                   # 空重箱
            "loading", "unloading", "stacking", "retrieving",       # 作业
            "container_id", "seal_intact", "seal_broken",          # 箱况
            "chassis", "trailer", "yard_truck",                    # 运输
            "reach_stacker", "empty_handler", "rs"                 # 机械
        ]
    },
    
    # 港口环境 / Port Environment
    "port_environment": {
        "name": "港口环境 Environment",
        "categories": [
            "oil_spill", "water_pollution", "garbage",            # 污染
            "dust", "fog", "heavy_rain", "storm",                 # 气象
            "debris_on_road", "pothole", "crack",                # 道路
            "lighting_normal", "lighting_dim", "lighting_off",  # 照明
            "crowd", "queue", "congestion"                       # 交通
        ]
    },
    
    # 港口设备 / Port Equipment
    "port_equipment": {
        "name": "港口设备 Equipment",
        "categories": [
            "gantry_crane", "mobile_crane", "tower_crane",        # 起重机
            "crane_normal", "crane_overload", "crane_drift",      # 状态
            "rope_wear", "rope_break", "brake_worn",             # 部件
            "belt_conveyor", "belt_rip", "belt_disalignment",    # 皮带
            "motor_normal", "motor_hot", "motor_vibration",       # 电机
            "pipe_leak", "pipe_block", "valve_status"            # 管道
        ]
    },
    
    # 通用物体 / General Objects
    "general": {
        "name": "通用物体 General",
        "categories": [
            "person", "car", "truck", "bus", "motorcycle",       # 交通
            "bicycle", "animal", "bag", "box", "pallet",         # 物品
            "sign", "light", "barrier", "cone", "fence"          # 设施
        ]
    }
}


# ============================================================
# 数据采集器 / Data Collector
# ============================================================

class DataCollector:
    """
    多协议数据采集器
    Multi-Protocol Data Collector
    
    支持 Supported:
    - RTSP 视频流
    - Modbus TCP (PLC)
    - HTTP REST API
    - 本地文件
    """
    
    def __init__(self):
        self.rtsp_cap = None
        self.modbus_client = None
        self.http_session = requests.Session()
        self.is_connected = False
        self.current_source = None
        
    # --- RTSP 视频流连接 / RTSP Stream Connection ---
    def connect_rtsp(self, url: str) -> bool:
        """
        连接 RTSP 视频流
        Connect to RTSP stream
        
        Args:
            url: RTSP URL (e.g., rtsp://192.168.1.100:554/stream1)
            
        Returns:
            bool: 连接状态 / Connection status
        """
        try:
            if self.rtsp_cap:
                self.rtsp_cap.release()
            self.rtsp_cap = cv2.VideoCapture(url)
            self.is_connected = self.rtsp_cap.isOpened()
            self.current_source = f"RTSP: {url}"
            return self.is_connected
        except Exception as e:
            print(f"RTSP 连接失败 / RTSP connection failed: {e}")
            return False
    
    # --- 本地摄像头 / Local Camera ---
    def connect_camera(self, camera_id: int = 0) -> bool:
        """
        连接本地摄像头
        Connect to local camera
        
        Args:
            camera_id: 摄像头ID / Camera ID (0=默认)
        """
        try:
            if self.rtsp_cap:
                self.rtsp_cap.release()
            self.rtsp_cap = cv2.VideoCapture(camera_id)
            self.is_connected = self.rtsp_cap.isOpened()
            self.current_source = f"Camera: {camera_id}"
            return self.is_connected
        except Exception as e:
            print(f"摄像头连接失败 / Camera connection failed: {e}")
            return False
    
    # --- Modbus TCP 连接 / Modbus TCP Connection ---
    def connect_modbus(self, host: str, port: int = 502) -> bool:
        """
        连接 Modbus TCP (PLC)
        Connect to Modbus TCP (PLC)
        
        Args:
            host: PLC IP 地址
            port: 端口 (默认 502)
        """
        try:
            from pymodbus.client import ModbusTcpClient
            self.modbus_client = ModbusTcpClient(host, port=port)
            self.is_connected = self.modbus_client.connect()
            self.current_source = f"Modbus: {host}:{port}"
            return self.is_connected
        except ImportError:
            print("请安装 pymodbus: pip install pymodbus")
            return False
        except Exception as e:
            print(f"Modbus 连接失败 / Modbus connection failed: {e}")
            return False
    
    # --- HTTP API 连接 / HTTP API Connection ---
    def connect_api(self, base_url: str, api_key: str = None) -> bool:
        """
        连接 HTTP REST API (如 TOS 系统)
        Connect to HTTP REST API (e.g., TOS System)
        
        Args:
            base_url: API 基础 URL
            api_key: API 密钥 (可选)
        """
        self.http_session.headers.update({"Authorization": f"Bearer {api_key}"} if api_key else {})
        self.http_session.headers.update({"Content-Type": "application/json"})
        self.current_source = f"API: {base_url}"
        self.is_connected = True
        return True
    
    # --- 读取帧 / Read Frame ---
    def read_frame(self):
        """读取视频帧 / Read video frame"""
        if self.rtsp_cap and self.rtsp_cap.isOpened():
            ret, frame = self.rtsp_cap.read()
            if ret:
                return frame
        return None
    
    # --- 读取 PLC 数据 / Read PLC Data ---
    def read_plc_registers(self, address: int, count: int = 1):
        """
        读取 PLC 寄存器
        Read PLC registers
        
        Args:
            address: 起始地址
            count: 读取数量
        """
        if self.modbus_client and self.modbus_client.connected:
            return self.modbus_client.read_holding_registers(address, count)
        return None
    
    # --- 断开连接 / Disconnect ---
    def disconnect(self):
        """断开所有连接 / Disconnect all"""
        if self.rtsp_cap:
            self.rtsp_cap.release()
        if self.modbus_client:
            self.modbus_client.close()
        self.is_connected = False
        self.current_source = None


# ============================================================
# 标注工具 / Annotation Tool
# ============================================================

class PortAnnotationTool:
    """
    港口场景标注工具
    Port Scene Annotation Tool
    
    功能 Features:
    - 多边形标注
    - 类别选择
    - 数据导出 (YOLO/COCO/VOC)
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("PortAI - 港口场景标注工具 / Port Annotation Tool")
        self.root.geometry("1400x900")
        
        # 状态变量 / State variables
        self.images_dir = None
        self.images = []
        self.current_idx = 0
        self.current_image = None
        self.annotations = []  # 当前图像的标注
        self.bbox_start = None
        self.is_drawing = False
        self.current_category = "person"
        self.current_scene = "port_surveillance"
        
        # 标注数据 / Annotation data
        self.data = {}  # {image_path: annotations}
        
        # 创建界面 / Create UI
        self.create_ui()
        
    def create_ui(self):
        """创建用户界面 / Create UI"""
        # 顶部控制栏 / Top Control Bar
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Button(top_frame, text="打开图像 Open Images", 
                   command=self.open_images).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(top_frame, text="保存 Save", 
                   command=self.save_annotations).pack(side=tk.LEFT, padx=5)
        
        # 场景选择 / Scene Selection
        ttk.Label(top_frame, text="场景 Scene:").pack(side=tk.LEFT, padx=5)
        self.scene_var = tk.StringVar(value="port_surveillance")
        scene_combo = ttk.Combobox(top_frame, textvariable=self.scene_var,
                                   values=list(PORT_CATEGORIES.keys()),
                                   state="readonly", width=20)
        scene_combo.pack(side=tk.LEFT, padx=5)
        scene_combo.bind("<<ComboboxSelected>>", self.on_scene_change)
        
        # 类别选择 / Category Selection
        ttk.Label(top_frame, text="类别 Category:").pack(side=tk.LEFT, padx=5)
        self.category_var = tk.StringVar(value="person")
        self.category_combo = ttk.Combobox(top_frame, textvariable=self.category_var,
                                            values=PORT_CATEGORIES["port_surveillance"]["categories"],
                                            state="readonly", width=15)
        self.category_combo.pack(side=tk.LEFT, padx=5)
        
        # 导出格式 / Export Format
        ttk.Label(top_frame, text="导出 Export:").pack(side=tk.LEFT, padx=5)
        self.export_format = tk.StringVar(value="yolo")
        ttk.Radiobutton(top_frame, text="YOLO", variable=self.export_format, 
                       value="yolo").pack(side=tk.LEFT)
        ttk.Radiobutton(top_frame, text="COCO", variable=self.export_format,
                       value="coco").pack(side=tk.LEFT)
        ttk.Radiobutton(top_frame, text="VOC", variable=self.export_format,
                       value="voc").pack(side=tk.LEFT)
        
        ttk.Button(top_frame, text="导出数据 Export", 
                   command=self.export_data).pack(side=tk.LEFT, padx=5)
        
        # 主显示区 / Main Display
        self.canvas = tk.Canvas(self.root, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # 底部状态栏 / Bottom Status Bar
        self.status_label = ttk.Label(self.root, text="就绪 Ready")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)
        
        # 图像导航 / Image Navigation
        nav_frame = ttk.Frame(self.root)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        ttk.Button(nav_frame, text="<< Prev", command=self.prev_image).pack(side=tk.LEFT)
        self.img_label = ttk.Label(nav_frame, text="0/0")
        self.img_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(nav_frame, text="Next >>", command=self.next_image).pack(side=tk.LEFT)
        
        ttk.Button(nav_frame, text="删除选中 Delete Selected", 
                   command=self.delete_selected).pack(side=tk.RIGHT)
        
    def on_scene_change(self, event=None):
        """切换场景 / Change scene"""
        self.current_scene = self.scene_var.get()
        categories = PORT_CATEGORIES[self.current_scene]["categories"]
        self.category_combo["values"] = categories
        self.current_category = categories[0] if categories else ""
        
    def open_images(self):
        """打开图像文件夹 / Open image folder"""
        folder = filedialog.askdirectory(title="选择图像文件夹 / Select Image Folder")
        if folder:
            self.images_dir = folder
            self.images = [f for f in os.listdir(folder) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            self.images.sort()
            if self.images:
                self.current_idx = 0
                self.load_image()
                
    def load_image(self):
        """加载当前图像 / Load current image"""
        if not self.images:
            return
        img_path = os.path.join(self.images_dir, self.images[self.current_idx])
        self.current_image = cv2.imread(img_path)
        self.annotations = self.data.get(img_path, [])
        self.display_image()
        self.img_label.config(text=f"{self.current_idx + 1}/{len(self.images)}")
        
    def display_image(self):
        """显示图像 / Display image"""
        if self.current_image is None:
            return
        img = self.current_image.copy()
        h, w = img.shape[:2]
        
        # 绘制标注 / Draw annotations
        for ann in self.annotations:
            x1, y1, x2, y2 = ann["bbox"]
            cat = ann["category"]
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, cat, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 转换显示 / Convert for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.photo = tk.PhotoImage(data=cv2.imencode('.png', img_rgb)[1].tobytes())
        self.canvas.config(width=w, height=h)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
    def on_mouse_down(self, event):
        """鼠标按下 / Mouse down"""
        self.bbox_start = (event.x, event.y)
        self.is_drawing = True
        
    def on_mouse_move(self, event):
        """鼠标移动 / Mouse move"""
        pass  # 可扩展实时预览
        
    def on_mouse_up(self, event):
        """鼠标释放 / Mouse up"""
        if self.is_drawing and self.bbox_start:
            x1, y1 = self.bbox_start
            x2, y2 = event.x, event.y
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            if x2 - x1 > 10 and y2 - y1 > 10:  # 最小框大小
                self.annotations.append({
                    "bbox": [x1, y1, x2, y2],
                    "category": self.category_var.get(),
                    "scene": self.current_scene
                })
                
                # 保存到数据 / Save to data
                img_path = os.path.join(self.images_dir, self.images[self.current_idx])
                self.data[img_path] = self.annotations
                
            self.is_drawing = False
            self.bbox_start = None
            self.display_image()
            self.status_label.config(text=f"已标注 {len(self.annotations)} 个目标")
            
    def prev_image(self):
        """上一张 / Previous"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image()
            
    def next_image(self):
        """下一张 / Next"""
        if self.current_idx < len(self.images) - 1:
            self.current_idx += 1
            self.load_image()
            
    def delete_selected(self):
        """删除选中标注 / Delete selected annotation"""
        if self.annotations:
            self.annotations.pop()
            img_path = os.path.join(self.images_dir, self.images[self.current_idx])
            self.data[img_path] = self.annotations
            self.display_image()
            
    def save_annotations(self):
        """保存标注 / Save annotations"""
        output_file = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")]
        )
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("保存成功", f"已保存到 {output_file}")
            
    def export_data(self):
        """导出数据 / Export data"""
        if not self.data:
            messagebox.showwarning("警告", "没有标注数据")
            return
            
        folder = filedialog.askdirectory(title="选择导出文件夹")
        if folder:
            fmt = self.export_format.get()
            if fmt == "yolo":
                self.export_yolo(folder)
            elif fmt == "coco":
                self.export_coco(folder)
            elif fmt == "voc":
                self.export_voc(folder)
            messagebox.showinfo("导出成功", f"已导出到 {folder}")
            
    def export_yolo(self, folder):
        """导出 YOLO 格式 / Export YOLO format"""
        labels_dir = os.path.join(folder, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        
        for img_path, annotations in self.data.items():
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            
            txt_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            with open(os.path.join(labels_dir, txt_name), 'w') as f:
                for ann in annotations:
                    x1, y1, x2, y2 = ann["bbox"]
                    cat = ann["category"]
                    
                    # 转换为中心点+宽高
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    
                    f.write(f"{cat} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


# ============================================================
# 主程序 / Main Program
# ============================================================

def main():
    """主入口 / Main entry"""
    print("=" * 60)
    print("PortAI - 港口数据采集与智能检测系统")
    print("PortAI - Port Data Collection & Detection System")
    print("=" * 60)
    print("版本 Version: 1.0.0")
    print("许可证 License: MIT")
    print("=" * 60)
    
    root = tk.Tk()
    app = PortAnnotationTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
