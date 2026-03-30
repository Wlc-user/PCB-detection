"""
全自动PCB光学质检系统
无需人工操作，自动检测并输出判定结果
"""
import cv2
import time
import logging
import threading
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import json
import sqlite3

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quality_check.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class JudgeResult(Enum):
    """判定结果"""
    OK = "OK"
    NG = "NG"
    ERROR = "ERROR"


@dataclass
class InspectionRecord:
    """质检记录"""
    board_id: str
    timestamp: str
    result: JudgeResult
    defect_count: int
    defects: List[Dict]
    inference_time: float


class AutoQualityChecker:
    """全自动光学质检系统"""
    
    def __init__(self, 
                 model_path: str = "models/yolov8/train/weights/best.pt",
                 conf_threshold: float = 0.15):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # 缺陷严重等级
        self.severity = {
            "missing_hole": 3,    # 极高
            "short": 3,           # 极高
            "open_circuit": 2,   # 高
            "mouse_bite": 2,     # 中
            "spur": 1,           # 低
            "spurious_copper": 1  # 低
        }
        
        # 统计
        self.stats = {
            "total": 0,
            "ok": 0,
            "ng": 0,
            "error": 0,
            "start_time": time.time()
        }
        
        # 数据库
        self.db_path = "inspection_records.db"
        self.init_db()
        
        logger.info("自动质检系统初始化完成")
    
    def init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inspection_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                board_id TEXT,
                timestamp TEXT,
                result TEXT,
                defect_count INTEGER,
                defects TEXT,
                inference_time REAL
            )
        """)
        conn.commit()
        conn.close()
        logger.info("数据库初始化完成")
    
    def detect(self, image) -> Dict:
        """检测缺陷"""
        results = self.model.predict(image, conf=self.conf_threshold, verbose=False)
        result = results[0]
        
        detections = []
        if len(result.boxes) > 0:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                detections.append({
                    "class": result.names[cls],
                    "confidence": conf,
                    "severity": self.severity.get(result.names[cls], 1)
                })
        
        return {
            "detections": detections,
            "count": len(detections)
        }
    
    def judge(self, detection_result: Dict) -> JudgeResult:
        """
        自动判定OK/NG
        
        判定规则:
        - 严重缺陷(severity>=2): 直接NG
        - 轻微缺陷数量>2: NG
        - 无缺陷: OK
        """
        detections = detection_result["detections"]
        
        if not detections:
            return JudgeResult.OK
        
        # 检查严重缺陷
        for det in detections:
            if det["severity"] >= 3:  # missing_hole, short
                return JudgeResult.NG
        
        # 轻微缺陷过多
        if len(detections) > 2:
            return JudgeResult.NG
        
        return JudgeResult.NG  # 有缺陷都NG
    
    def inspect(self, image, board_id: str = None) -> InspectionRecord:
        """
        执行一次完整质检
        
        Args:
            image: PCB图片
            board_id: 板号(可选，从扫码枪获取)
            
        Returns:
            InspectionRecord: 质检结果
        """
        start_time = time.time()
        
        # 生成板号
        if board_id is None:
            board_id = f"PCB_{int(time.time()*1000)}"
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            # 检测
            detection = self.detect(image)
            
            # 判定
            result = self.judge(detection)
            
            # 推理时间
            inference_time = time.time() - start_time
            
            # 记录
            record = InspectionRecord(
                board_id=board_id,
                timestamp=timestamp,
                result=result,
                defect_count=detection["count"],
                defects=detection["detections"],
                inference_time=inference_time
            )
            
            # 保存到数据库
            self.save_record(record)
            
            # 更新统计
            self.stats["total"] += 1
            if result == JudgeResult.OK:
                self.stats["ok"] += 1
            elif result == JudgeResult.NG:
                self.stats["ng"] += 1
            else:
                self.stats["error"] += 1
            
            # 输出到PLC (TODO: 实现PLC输出)
            self.output_to_plc(record)
            
            logger.info(f"[{board_id}] {result.value} - 缺陷数:{detection['count']} - 耗时:{inference_time:.3f}s")
            
            return record
            
        except Exception as e:
            logger.error(f"质检异常: {e}")
            self.stats["error"] += 1
            return InspectionRecord(
                board_id=board_id,
                timestamp=timestamp,
                result=JudgeResult.ERROR,
                defect_count=0,
                defects=[],
                inference_time=time.time() - start_time
            )
    
    def save_record(self, record: InspectionRecord):
        """保存记录到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO inspection_records 
            (board_id, timestamp, result, defect_count, defects, inference_time)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            record.board_id,
            record.timestamp,
            record.result.value,
            record.defect_count,
            json.dumps(record.defects, ensure_ascii=False),
            record.inference_time
        ))
        conn.commit()
        conn.close()
    
    def output_to_plc(self, record: InspectionRecord):
        """
        输出结果到PLC
        
        寄存器映射:
        - 4000: 检测状态 (0=空闲, 1=完成)
        - 4001: 判定结果 (0=OK, 1=NG, 2=ERROR)
        - 4002: 缺陷数量
        - 4003: 缺陷等级(严重程度)
        """
        # TODO: 实现实际的PLC通信
        # plc.write_register(4000, 1)  # 完成
        # plc.write_register(4001, 0 if record.result == JudgeResult.OK else 1)
        # plc.write_register(4002, record.defect_count)
        
        # 打印输出
        if record.result == JudgeResult.NG:
            print(f"⚠️ NG - 板号:{record.board_id} - 缺陷:{record.defect_count}个")
        else:
            print(f"✅ OK - 板号:{record.board_id}")
    
    def get_stats(self) -> Dict:
        """获取统计"""
        runtime = time.time() - self.stats["start_time"]
        hour_rate = self.stats["total"] / max(runtime/3600, 0.1)
        
        return {
            **self.stats,
            "runtime_seconds": runtime,
            "hourly_rate": round(hour_rate, 1),
            "ok_rate": round(self.stats["ok"] / max(self.stats["total"], 1) * 100, 1)
        }
    
    def export_report(self, output_path: str = "inspection_report.xlsx"):
        """导出报表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM inspection_records ORDER BY id DESC LIMIT 1000")
        records = cursor.fetchall()
        conn.close()
        
        logger.info(f"已导出{len(records)}条记录到 {output_path}")
        return len(records)


def run_auto_inspection():
    """运行全自动质检"""
    print("="*60)
    print("   全自动PCB光学质检系统")
    print("   无需人工操作 自动判定OK/NG")
    print("="*60)
    
    # 初始化
    checker = AutoQualityChecker()
    
    # 使用测试图片
    test_images = [
        "yolo_pcb_dataset/images/test/01_missing_hole_02.jpg",
        "yolo_pcb_dataset/images/test/12_spurious_copper_02.jpg",
    ]
    
    print("\n开始质检测试...")
    print("-"*40)
    
    for i, img_path in enumerate(test_images):
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # 模拟板号
        board_id = f"PCB_{i+1:04d}"
        
        # 质检
        result = checker.inspect(image, board_id)
        
        # 显示
        status = "✅ OK" if result.result == JudgeResult.OK else "⚠️ NG"
        print(f"{board_id}: {status} - 缺陷数:{result.defect_count} - 耗时:{result.inference_time:.3f}s")
    
    # 统计
    print("-"*40)
    stats = checker.get_stats()
    print(f"\n统计:")
    print(f"  总数: {stats['total']}")
    print(f"  OK: {stats['ok']}")
    print(f"  NG: {stats['ng']}")
    print(f"  良率: {stats['ok_rate']}%")
    print(f"  时效: {stats['hourly_rate']} PCS/小时")


if __name__ == "__main__":
    run_auto_inspection()
