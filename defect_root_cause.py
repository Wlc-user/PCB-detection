"""
多工序流水线缺陷根因定位系统
结合YOLOv8缺陷检测 + DSSM双塔匹配

功能：
1. YOLOv8实时缺陷检测
2. 缺陷特征提取
3. DSSM双塔匹配定位根因
4. 工序异常预警
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
from ultralytics import YOLO
from typing import List, Dict, Tuple
import json
from dataclasses import dataclass
from enum import Enum


class ProcessStage(str, Enum):
    """PCB生产工序"""
    # 表面处理工序
    SURFACE_CLEANING = "表面清洗"           # 清洗、除油
    COPPER_PLATING = "沉铜"                # 化学沉铜
    PATTERN_TRANSFER = "图形转移"           # 曝光、显影
    ETCHING = "蚀刻"                       # 蚀刻铜箔
    SOLDER_MASK = "阻焊"                   # 绿油涂覆
    SILKSCREEN = "丝印"                    # 字符丝印
    SURFACE_FINISH = "表面处理"            # 沉金/OSP/喷锡
    
    # 组装工序
    SMT = "SMT贴片"                        # 贴片机贴装
    REFLOW = "回流焊"                      # 回流焊接
    WAVE_SOLDERING = "波峰焊"              # 波峰焊接
    HAND_SOLDERING = "手工焊接"             # 人工补焊
    CLEANING = "清洗"                      # 清洗助焊剂
    
    # 检测工序
    AOI = "AOI检测"                        # 自动光学检测
    X_RAY = "X-Ray检测"                   # X射线检测
    ICT = "ICT测试"                        # 在线测试
    FUNCTIONAL_TEST = "功能测试"           # 功能测试


# 缺陷类型与可能工序的映射关系
DEFECT_PROCESS_MAPPING = {
    # 缺失类缺陷 → 可能是图形转移/蚀刻/电镀问题
    "missing_hole": {
        "primary": [ProcessStage.COPPER_PLATING, ProcessStage.PATTERN_TRANSFER],
        "secondary": [ProcessStage.ETCHING, ProcessStage.AOI],
        "symptoms": "孔洞区域铜层缺失或孔径偏小"
    },
    
    # 鼠咬缺陷 → 蚀刻工序问题
    "mouse_bite": {
        "primary": [ProcessStage.ETCHING, ProcessStage.PATTERN_TRANSFER],
        "secondary": [ProcessStage.COPPER_PLATING, ProcessStage.AOI],
        "symptoms": "走线边缘出现缺口或凹陷"
    },
    
    # 开路缺陷 → 蚀刻/电镀/贴片问题
    "open_circuit": {
        "primary": [ProcessStage.ETCHING, ProcessStage.SMT, ProcessStage.REFLOW],
        "secondary": [ProcessStage.COPPER_PLATING, ProcessStage.AOI],
        "symptoms": "走线断开，电气连接中断"
    },
    
    # 短路缺陷 → 蚀刻/阻焊/回流焊问题
    "short": {
        "primary": [ProcessStage.ETCHING, ProcessStage.SOLDER_MASK, ProcessStage.REFLOW],
        "secondary": [ProcessStage.SMT, ProcessStage.AOI, ProcessStage.X_RAY],
        "symptoms": "相邻走线意外连接"
    },
    
    # 毛刺缺陷 → 蚀刻/切割问题
    "spur": {
        "primary": [ProcessStage.ETCHING, ProcessStage.PATTERN_TRANSFER],
        "secondary": [ProcessStage.AOI],
        "symptoms": "走线边缘出现多余铜刺"
    },
    
    # 多余铜缺陷 → 蚀刻/电镀问题
    "spurious_copper": {
        "primary": [ProcessStage.ETCHING, ProcessStage.COPPER_PLATING],
        "secondary": [ProcessStage.PATTERN_TRANSFER, ProcessStage.AOI],
        "symptoms": "非设计区域出现铜箔残留"
    },
    
    # 焊点缺陷 → 贴片/焊接问题
    "solder_bridge": {
        "primary": [ProcessStage.REFLOW, ProcessStage.WAVE_SOLDERING],
        "secondary": [ProcessStage.SMT, ProcessStage.AOI],
        "symptoms": "焊点之间出现桥连"
    },
    
    # 虚焊缺陷 → 焊接问题
    "cold_solder": {
        "primary": [ProcessStage.REFLOW, ProcessStage.WAVE_SOLDERING],
        "secondary": [ProcessStage.HAND_SOLDERING, ProcessStage.X_RAY],
        "symptoms": "焊点光泽度差，连接不牢"
    },
    
    # 元件偏移 → 贴片问题
    "component_shift": {
        "primary": [ProcessStage.SMT, ProcessStage.REFLOW],
        "secondary": [ProcessStage.AOI],
        "symptoms": "元件位置偏离设计位置"
    },
    
    # 元件缺失 → 贴片问题
    "missing_component": {
        "primary": [ProcessStage.SMT],
        "secondary": [ProcessStage.AOI, ProcessStage.FUNCTIONAL_TEST],
        "symptoms": "指定位置没有元件"
    },
}


class DSSMProcessor(nn.Module):
    """
    DSSM双塔模型
    - 缺陷塔: 将缺陷特征映射到语义空间
    - 工序塔: 将工序特征映射到语义空间
    - 通过余弦相似度匹配找到最可能的问题工序
    """
    
    def __init__(self, defect_dim=128, process_dim=64, hidden_dim=128):
        super().__init__()
        
        # 缺陷特征塔 (Defect Tower)
        self.defect_tower = nn.Sequential(
            nn.Linear(defect_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, process_dim),
        )
        
        # 工序特征塔 (Process Tower)
        self.process_tower = nn.Sequential(
            nn.Linear(process_dim * 3, hidden_dim),  # 3个特征: 类型+位置+严重程度
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, process_dim),
        )
        
        # 缺陷类型编码
        self.defect_type_embedding = nn.Embedding(20, 64)
        
        # 位置编码
        self.position_encoder = nn.Linear(4, 32)  # x,y,w,h
        
        # 严重程度编码
        self.severity_encoder = nn.Linear(1, 32)
    
    def encode_defect(self, defect_features: torch.Tensor) -> torch.Tensor:
        """编码缺陷特征"""
        return self.defect_tower(defect_features)
    
    def encode_process(self, process_features: torch.Tensor) -> torch.Tensor:
        """编码工序特征"""
        return self.process_tower(process_features)
    
    def forward(self, defect_feat, process_feat):
        """前向传播，计算相似度"""
        d_out = self.encode_defect(defect_feat)
        p_out = self.encode_process(process_feat)
        
        # 余弦相似度
        cos_sim = nn.functional.cosine_similarity(d_out, p_out, dim=-1)
        return cos_sim


class RootCauseAnalyzer:
    """根因分析器"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化DSSM模型
        self.dssm = DSSMProcessor(
            defect_dim=128,
            process_dim=64,
            hidden_dim=128
        ).to(self.device)
        
        self.dssm.eval()
        
        # 初始化工序特征库
        self.process_features = self._build_process_feature_library()
        
        # 加载YOLOv8模型
        print("加载YOLOv8缺陷检测模型...")
        self.yolo = YOLO("models/yolov8/train/weights/best.pt")
        
        # 缺陷类别
        self.defect_classes = [
            'missing_hole', 'mouse_bite', 'open_circuit',
            'short', 'spur', 'spurious_copper'
        ]
    
    def _build_process_feature_library(self) -> Dict:
        """构建工序特征库"""
        # 每个工序的特征向量 (类型编码 + 历史缺陷权重 + 风险因子)
        process_features = {}
        
        for stage in ProcessStage:
            # 基于工序类型和历史数据生成特征
            base_features = self._get_process_features(stage)
            process_features[stage.value] = base_features
        
        return process_features
    
    def _get_process_features(self, stage: ProcessStage) -> np.ndarray:
        """获取工序特征"""
        # 特征: [风险等级, 缺陷敏感性, 历史问题频率, 设备依赖度]
        features = {
            ProcessStage.SURFACE_CLEANING: [0.3, 0.4, 0.2, 0.3],
            ProcessStage.COPPER_PLATING: [0.7, 0.8, 0.5, 0.8],
            ProcessStage.PATTERN_TRANSFER: [0.6, 0.7, 0.4, 0.7],
            ProcessStage.ETCHING: [0.8, 0.9, 0.6, 0.8],
            ProcessStage.SOLDER_MASK: [0.4, 0.5, 0.3, 0.5],
            ProcessStage.SILKSCREEN: [0.2, 0.3, 0.2, 0.3],
            ProcessStage.SURFACE_FINISH: [0.5, 0.6, 0.4, 0.6],
            ProcessStage.SMT: [0.7, 0.8, 0.6, 0.9],
            ProcessStage.REFLOW: [0.8, 0.9, 0.7, 0.9],
            ProcessStage.WAVE_SOLDERING: [0.7, 0.8, 0.5, 0.8],
            ProcessStage.HAND_SOLDERING: [0.4, 0.5, 0.3, 0.4],
            ProcessStage.CLEANING: [0.3, 0.4, 0.2, 0.3],
            ProcessStage.AOI: [0.1, 0.2, 0.1, 0.5],
            ProcessStage.X_RAY: [0.1, 0.2, 0.1, 0.6],
            ProcessStage.ICT: [0.2, 0.3, 0.2, 0.5],
            ProcessStage.FUNCTIONAL_TEST: [0.2, 0.3, 0.2, 0.5],
        }
        
        return np.array(features.get(stage, [0.5, 0.5, 0.5, 0.5]))
    
    def detect_and_analyze(self, image_path: str, conf_threshold: float = 0.15) -> Dict:
        """
        检测缺陷并分析根因
        
        Args:
            image_path: 图片路径
            conf_threshold: 置信度阈值
            
        Returns:
            检测结果 + 根因分析
        """
        # 1. YOLOv8缺陷检测
        results = self.yolo.predict(image_path, conf=conf_threshold, verbose=False)
        
        detections = []
        root_causes = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                defect_type = self.defect_classes[cls_id] if cls_id < len(self.defect_classes) else "unknown"
                
                # 2. 提取缺陷特征
                defect_feature = self._extract_defect_feature(defect_type, xyxy, conf)
                
                # 3. DSSM根因分析
                cause_analysis = self._analyze_root_cause(defect_type, defect_feature, xyxy)
                
                detection = {
                    "defect_type": defect_type,
                    "confidence": round(conf, 3),
                    "bbox": [float(x) for x in xyxy],
                    "root_cause": cause_analysis
                }
                
                detections.append(detection)
                root_causes.append(cause_analysis)
        
        # 4. 汇总工序异常
        process_alerts = self._aggregate_process_alerts(root_causes)
        
        return {
            "image": image_path,
            "total_defects": len(detections),
            "detections": detections,
            "process_alerts": process_alerts,
            "recommendation": self._generate_recommendation(process_alerts)
        }
    
    def _extract_defect_feature(self, defect_type: str, bbox: np.ndarray, confidence: float) -> np.ndarray:
        """提取缺陷特征"""
        # 特征: [缺陷类型编码, 位置编码, 尺寸编码, 置信度]
        
        # 类型编码 (one-hot简化)
        type_code = self.defect_classes.index(defect_type) if defect_type in self.defect_classes else 0
        
        # 位置编码 (归一化)
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2 / 1920  # 假设最大宽度
        center_y = (y1 + y2) / 2 / 1080  # 假设最大高度
        width = (x2 - x1) / 1920
        height = (y2 - y1) / 1080
        
        # 严重程度 (基于尺寸和置信度)
        severity = (width * height) * confidence
        
        feature = np.array([
            type_code / 10,
            center_x, center_y,
            width, height,
            confidence,
            severity
        ])
        
        return feature
    
    def _analyze_root_cause(self, defect_type: str, defect_feature: np.ndarray, bbox: np.ndarray) -> Dict:
        """分析缺陷根因"""
        
        if defect_type not in DEFECT_PROCESS_MAPPING:
            return {
                "defect_type": defect_type,
                "possible_processes": [],
                "primary_cause": "未知",
                "confidence": 0.0
            }
        
        mapping = DEFECT_PROCESS_MAPPING[defect_type]
        
        # 计算每个工序的匹配分数
        process_scores = []
        
        for stage in ProcessStage:
            score = 0.0
            
            # 主要工序加分
            if stage in mapping["primary"]:
                score += 0.6
            # 次要工序加分
            elif stage in mapping["secondary"]:
                score += 0.3
            
            # 工序特征匹配
            proc_feat = self.process_features.get(stage.value, np.array([0.5, 0.5, 0.5, 0.5]))
            
            # 基于位置计算额外分数
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            
            # 大面积缺陷可能是蚀刻/电镀问题
            if area > 50000 and stage in [ProcessStage.ETCHING, ProcessStage.COPPER_PLATING]:
                score += 0.2
            
            # 小面积缺陷可能是SMT问题
            if area < 5000 and stage in [ProcessStage.SMT, ProcessStage.REFLOW]:
                score += 0.2
            
            if score > 0:
                process_scores.append({
                    "process": stage.value,
                    "score": round(score, 3),
                    "is_primary": stage in mapping["primary"]
                })
        
        # 排序
        process_scores.sort(key=lambda x: x["score"], reverse=True)
        
        # 取前3个最可能的工序
        top_processes = process_scores[:3]
        
        return {
            "defect_type": defect_type,
            "symptoms": mapping["symptoms"],
            "possible_processes": top_processes,
            "primary_cause": top_processes[0]["process"] if top_processes else "未知",
            "confidence": top_processes[0]["score"] if top_processes else 0.0
        }
    
    def _aggregate_process_alerts(self, root_causes: List[Dict]) -> List[Dict]:
        """汇总工序异常预警"""
        process_counts = {}
        
        for cause in root_causes:
            for proc in cause.get("possible_processes", []):
                proc_name = proc["process"]
                if proc_name not in process_counts:
                    process_counts[proc_name] = {
                        "process": proc_name,
                        "count": 0,
                        "total_score": 0.0,
                        "defects": []
                    }
                
                process_counts[proc_name]["count"] += 1
                process_counts[proc_name]["total_score"] += proc["score"]
                process_counts[proc_name]["defects"].append(cause["defect_type"])
        
        # 计算平均分数并排序
        alerts = []
        for proc_name, data in process_counts.items():
            avg_score = data["total_score"] / data["count"]
            alerts.append({
                "process": proc_name,
                "defect_count": data["count"],
                "risk_score": round(avg_score * data["count"], 3),
                "defect_types": list(set(data["defects"]))
            })
        
        alerts.sort(key=lambda x: x["risk_score"], reverse=True)
        
        return alerts
    
    def _generate_recommendation(self, process_alerts: List[Dict]) -> str:
        """生成处理建议"""
        if not process_alerts:
            return "未检测到缺陷，工序正常运行"
        
        top_alert = process_alerts[0]
        
        recommendations = {
            "蚀刻": "检查蚀刻机药水浓度、温度、传送速度，必要时更换蚀刻液",
            "沉铜": "检查沉铜槽药液配比、电流密度、是否需要更换",
            "SMT贴片": "检查贴片机吸嘴、真空度、贴装精度，核对物料",
            "回流焊": "检查回流焊温度曲线、预热区、焊接区温度设置",
            "AOI检测": "调整AOI检测阈值，检查光源和相机状态",
            "图形转移": "检查曝光机能量、菲林质量、显影液浓度",
            "波峰焊": "检查波峰高度、预热温度、锡炉温度",
        }
        
        proc_name = top_alert["process"]
        base_recommend = recommendations.get(proc_name, "检查该工序设备和工艺参数")
        
        return f"【{proc_name}】风险最高，建议: {base_recommend}"


def demo():
    """演示"""
    print("=" * 60)
    print("  多工序流水线缺陷根因定位系统")
    print("=" * 60)
    
    # 初始化
    analyzer = RootCauseAnalyzer()
    
    # 检测图片
    test_image = "yolo_pcb_dataset/images/test/01_missing_hole_02.jpg"
    
    print(f"\n检测图片: {test_image}")
    print("-" * 40)
    
    result = analyzer.detect_and_analyze(test_image)
    
    # 输出结果
    print(f"\n检测到 {result['total_defects']} 个缺陷:")
    for d in result["detections"]:
        print(f"  - {d['defect_type']}: 置信度={d['confidence']}")
        cause = d['root_cause']
        print(f"    根因: {cause['primary_cause']} (置信度: {cause['confidence']})")
    
    print("\n" + "=" * 40)
    print("工序异常预警:")
    print("=" * 40)
    for alert in result["process_alerts"]:
        print(f"  ⚠️ {alert['process']}: 风险指数={alert['risk_score']}, 缺陷数={alert['defect_count']}")
        print(f"     缺陷类型: {', '.join(alert['defect_types'])}")
    
    print("\n" + "=" * 40)
    print("处理建议:")
    print("=" * 40)
    print(f"  {result['recommendation']}")


if __name__ == "__main__":
    demo()
