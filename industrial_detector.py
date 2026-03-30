"""
工控级PCB缺陷检测系统
- YOLOv8实时检测
- PLC通信 (Modbus TCP)
- 看门狗保护
- 日志记录
"""
import cv2
import numpy as np
import time
import logging
import threading
from pathlib import Path
from ultralytics import YOLO
from typing import Optional, Dict, List
from plc_communication import DetectionPLC
import struct
import socket

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('detector.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IndustrialDetector:
    """工控级缺陷检测器"""
    
    def __init__(self, 
                 model_path: str = "models/yolov8/train/weights/best.pt",
                 conf_threshold: float = 0.15,
                 plc_ip: str = "192.168.1.100"):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.plc_ip = plc_ip
        
        # 加载模型
        logger.info(f"加载模型: {model_path}")
        self.model = YOLO(model_path)
        logger.info("模型加载成功")
        
        # 缺陷编码
        self.defect_codes = {
            0: "missing_hole",
            1: "mouse_bite", 
            2: "open_circuit",
            3: "short",
            4: "spur",
            5: "spurious_copper"
        }
        
        # PLC通信
        self.plc = DetectionPLC(plc_ip)
        self.plc_connected = False
        
        # 统计
        self.stats = {
            "total": 0,
            "defect": 0,
            "ok": 0,
            "error": 0
        }
        
    def connect_plc(self) -> bool:
        """连接PLC"""
        try:
            self.plc_connected = self.plc.plc.connect()
            if self.plc_connected:
                logger.info(f"PLC连接成功: {self.plc_ip}")
            else:
                logger.warning(f"PLC连接失败: {self.plc_ip}")
            return self.plc_connected
        except Exception as e:
            logger.error(f"PLC连接异常: {e}")
            return False
    
    def detect(self, image) -> Dict:
        """
        检测单帧图像
        
        Args:
            image: numpy数组 (BGR)
            
        Returns:
            {
                "success": bool,
                "detections": [{"class": str, "confidence": float, "bbox": [x1,y1,x2,y2]}],
                "inference_time": float,
                "total_defects": int
            }
        """
        start_time = time.time()
        
        try:
            # 推理
            results = self.model.predict(
                image,
                conf=self.conf_threshold,
                verbose=False
            )
            
            result = results[0]
            boxes = result.boxes
            
            # 解析结果
            detections = []
            if len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    detections.append({
                        "class": self.defect_codes.get(cls, "unknown"),
                        "confidence": conf,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)]
                    })
            
            inference_time = time.time() - start_time
            
            # 更新统计
            self.stats["total"] += 1
            if detections:
                self.stats["defect"] += 1
            else:
                self.stats["ok"] += 1
            
            return {
                "success": True,
                "detections": detections,
                "inference_time": inference_time,
                "total_defects": len(detections)
            }
            
        except Exception as e:
            logger.error(f"检测异常: {e}")
            self.stats["error"] += 1
            return {
                "success": False,
                "error": str(e),
                "inference_time": time.time() - start_time
            }
    
    def detect_and_output(self, image, output_plc: bool = True) -> Dict:
        """
        检测并输出到PLC
        """
        result = self.detect(image)
        
        if output_plc and self.plc_connected:
            try:
                self.plc.send_result(result)
            except Exception as e:
                logger.error(f"PLC输出失败: {e}")
        
        return result
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计"""
        self.stats = {"total": 0, "defect": 0, "ok": 0, "error": 0}
        logger.info("统计已重置")


class CameraGrabber:
    """工业相机采集器 (Demo使用USB/虚拟相机)"""
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        
    def open(self) -> bool:
        """打开相机"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.error(f"无法打开相机: {self.camera_index}")
            return False
        
        # 设置分辨率 (1280x720)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info(f"相机已打开: {self.camera_index}")
        return True
    
    def read(self):
        """读取一帧"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None
    
    def close(self):
        """关闭相机"""
        if self.cap:
            self.cap.release()
            logger.info("相机已关闭")


def run_industrial_demo():
    """运行工控Demo"""
    print("="*60)
    print("工控级PCB缺陷检测系统")
    print("="*60)
    
    # 创建检测器
    detector = IndustrialDetector(
        model_path="models/yolov8/train/weights/best.pt",
        conf_threshold=0.15,
        plc_ip="192.168.1.100"
    )
    
    # 连接PLC (可选)
    detector.connect_plc()
    
    # 打开相机 (或使用图片)
    camera = CameraGrabber(0)
    
    if not camera.open():
        print("无法打开相机，使用图片测试...")
        # 使用测试图片
        test_img = cv2.imread("yolo_pcb_dataset/images/test/01_missing_hole_02.jpg")
        if test_img is not None:
            result = detector.detect_and_output(test_img)
            print(f"\n检测结果:")
            print(f"  缺陷数: {result['total_defects']}")
            print(f"  推理时间: {result['inference_time']:.3f}s")
            for d in result['detections']:
                print(f"  - {d['class']}: {d['confidence']:.2f}")
        return
    
    # 实时检测循环
    print("\n按 'q' 退出, 按 's' 截图")
    
    while True:
        frame = camera.read()
        if frame is None:
            break
        
        # 检测
        result = detector.detect(frame)
        
        # 绘制结果
        for det in result.get("detections", []):
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 显示
        cv2.putText(frame, f"Defects: {result['total_defects']}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"FPS: {1/result['inference_time']:.1f}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Industrial Detector", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"capture_{int(time.time())}.jpg", frame)
            print("截图已保存")
    
    camera.close()
    cv2.destroyAllWindows()
    
    # 打印统计
    stats = detector.get_stats()
    print(f"\n统计: {stats}")


if __name__ == "__main__":
    run_industrial_demo()
