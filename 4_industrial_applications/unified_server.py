"""
REST API Perception + SLAM Server
支持视觉感知、SLAM定位、融合导航
"""

import cv2
import numpy as np
import base64
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from logistics_perception import LogisticsPerception
from unified_perception_slam import (
    UnifiedPerceptionSLAM, SimpleAGVPerception,
    create_unified_system, FusionLevel,
    Pose2D, SLAMType, SLAMConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AGV Perception + SLAM API", version="2.0.0")

# ============= 全局感知器 =============

# 统一感知+SLAM系统 (支持激光雷达)
unified_system = None

# 简单感知系统 (只用摄像头)
simple_system = None


def init_systems():
    global unified_system, simple_system
    
    logger.info("Initializing perception systems...")
    
    # 简单系统 - 快速启动
    try:
        simple_system = SimpleAGVPerception()
        logger.info("[OK] SimpleAGVPerception ready")
    except Exception as e:
        logger.error(f"[X] SimpleAGVPerception failed: {e}")
        simple_system = None
    
    # 统一系统 - 需要时间加载
    try:
        unified_system = create_unified_system("fusion")
        logger.info("[OK] UnifiedPerceptionSLAM ready")
    except Exception as e:
        logger.error(f"[X] UnifiedPerceptionSLAM failed: {e}")
        unified_system = None


# ============= 数据模型 =============

class DetectionRequest(BaseModel):
    """检测请求"""
    image_base64: str
    camera_id: str = "cam_0"
    include_slam: bool = False


class FusionRequest(BaseModel):
    """融合感知请求 (图像+激光)"""
    image_base64: str
    lidar_scan: Optional[List[float]] = None  # 360度激光数据
    odometry: Optional[List[float]] = None    # [vx, vy, vtheta]
    camera_id: str = "cam_0"


class NavigationRequest(BaseModel):
    """导航请求"""
    image_base64: str
    camera_id: str = "cam_0"
    target_distance: float = 2.0  # 前方安全距离


class DetectionResponse(BaseModel):
    """检测响应"""
    request_id: str
    camera_id: str
    timestamp: str
    processing_time_ms: float
    detections: List[Dict]
    scene_type: str
    pose: Optional[Dict] = None
    obstacles: Optional[List[Dict]] = None
    path_clear: bool = True


# ============= 路由 =============

@app.on_event("startup")
async def startup():
    init_systems()
    logger.info("[OK] Server startup complete")


@app.get("/")
async def root():
    return {
        "service": "AGV Perception + SLAM Server",
        "version": "2.0.0",
        "endpoints": {
            "POST /api/detect": "目标检测 (图像)",
            "POST /api/fusion": "融合感知 (图像+激光)",
            "POST /api/navigate": "导航感知",
            "GET /api/status": "系统状态",
            "GET /api/map": "获取地图",
            "GET /api/pose": "获取位姿",
            "GET /health": "健康检查",
            "GET /": "本页面"
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "systems": {
            "simple": simple_system is not None,
            "unified": unified_system is not None
        }
    }


@app.get("/api/status")
async def get_status():
    """获取系统状态"""
    status = {
        "server_time": datetime.now().isoformat(),
        "simple_system": None,
        "unified_system": None
    }
    
    if simple_system:
        status["simple_system"] = {"running": True}
    
    if unified_system:
        status["unified_system"] = unified_system.get_status()
    
    return status


@app.post("/api/detect", response_model=DetectionResponse)
async def detect(request: DetectionRequest):
    """
    目标检测API
    接收base64编码的图像，返回检测结果
    """
    import time
    import uuid
    request_id = str(uuid.uuid4())[:8]
    
    try:
        # 解码图像
        img_bytes = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # 感知
        if simple_system:
            result = simple_system.update(frame)
            
            return DetectionResponse(
                request_id=request_id,
                camera_id=request.camera_id,
                timestamp=datetime.now().isoformat(),
                processing_time_ms=result.processing_time_ms,
                detections=[
                    {
                        "class": obj.class_name,
                        "category": obj.category.value,
                        "bbox": list(obj.bbox),
                        "confidence": float(obj.confidence),
                        "track_id": obj.track_id
                    }
                    for obj in result.objects
                ],
                scene_type=result.scene_type.value,
                path_clear=result.path_clear
            )
        else:
            raise HTTPException(status_code=503, detail="Perception system not available")
    
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/fusion", response_model=DetectionResponse)
async def fusion(request: FusionRequest):
    """
    融合感知API (视觉+激光+SLAM)
    """
    import uuid
    request_id = str(uuid.uuid4())[:8]
    
    try:
        # 解码图像
        img_bytes = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # 转换激光数据
        scan = np.array(request.lidar_scan) if request.lidar_scan else None
        odometry = tuple(request.odometry) if request.odometry else None
        
        # 融合感知
        if unified_system:
            result = unified_system.update(
                frame=frame,
                scan=scan,
                odometry=odometry
            )
            
            return DetectionResponse(
                request_id=request_id,
                camera_id=request.camera_id,
                timestamp=datetime.now().isoformat(),
                processing_time_ms=result.processing_time_ms,
                detections=[
                    {
                        "class": obj.class_name,
                        "category": obj.category.value,
                        "bbox": list(obj.bbox),
                        "confidence": float(obj.confidence),
                        "track_id": obj.track_id
                    }
                    for obj in result.objects
                ],
                scene_type=result.scene_type.value,
                pose={
                    "x": float(result.pose.x),
                    "y": float(result.pose.y),
                    "theta": float(result.pose.theta)
                },
                obstacles=[
                    {
                        "x": o.position[0],
                        "y": o.position[1],
                        "size": list(o.size),
                        "category": o.category,
                        "dangerous": result._is_dangerous(o)
                    }
                    for o in result.obstacles
                ],
                path_clear=result.path_clear
            )
        elif simple_system:
            # 回退到简单感知
            result = simple_system.update(frame)
            
            return DetectionResponse(
                request_id=request_id,
                camera_id=request.camera_id,
                timestamp=datetime.now().isoformat(),
                processing_time_ms=result.processing_time_ms,
                detections=[
                    {
                        "class": obj.class_name,
                        "category": obj.category.value,
                        "bbox": list(obj.bbox),
                        "confidence": float(obj.confidence),
                        "track_id": obj.track_id
                    }
                    for obj in result.objects
                ],
                scene_type=result.scene_type.value,
                path_clear=result.path_clear
            )
        else:
            raise HTTPException(status_code=503, detail="No perception system available")
    
    except Exception as e:
        logger.error(f"Fusion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/navigate")
async def navigate(request: NavigationRequest):
    """
    导航感知API
    返回导航所需的障碍物和可通行区域
    """
    try:
        # 解码图像
        img_bytes = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        if simple_system:
            result = simple_system.update(frame)
            
            # 分析前方障碍物
            forward_obstacles = [
                o for o in result.obstacles
                if o.position[0] > 0  # 前方
            ]
            
            # 危险障碍物
            dangerous = [
                o for o in forward_obstacles
                if o.position[0] < request.target_distance and 
                   o.confidence > 0.7 and
                   o.category in ["person", "vehicle", "forklift"]
            ]
            
            return {
                "camera_id": request.camera_id,
                "timestamp": datetime.now().isoformat(),
                "forward_distance": request.target_distance,
                "obstacles": [
                    {
                        "x": o.position[0],
                        "y": o.position[1],
                        "category": o.category,
                        "dangerous": o in dangerous
                    }
                    for o in forward_obstacles
                ],
                "can_proceed": len(dangerous) == 0,
                "action": "stop" if dangerous else "move_forward"
            }
        else:
            raise HTTPException(status_code=503, detail="System not available")
    
    except Exception as e:
        logger.error(f"Navigate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pose")
async def get_pose():
    """获取当前位姿"""
    if unified_system:
        pose = unified_system.result.pose
        return {
            "x": float(pose.x),
            "y": float(pose.y),
            "theta": float(pose.theta),
            "timestamp": pose.timestamp
        }
    elif simple_system:
        pose = simple_system.prev_pose
        return {
            "x": float(pose.x),
            "y": float(pose.y),
            "theta": float(pose.theta),
            "timestamp": pose.timestamp
        }
    else:
        raise HTTPException(status_code=503, detail="No SLAM system available")


@app.get("/api/map")
async def get_map():
    """获取地图"""
    if unified_system and unified_system.result.global_map:
        m = unified_system.result.global_map
        return {
            "type": "occupancy_grid",
            "width": m.width,
            "height": m.height,
            "resolution": m.resolution,
            "origin": m.origin,
            "data": base64.b64encode(m.data.tobytes()).decode()
        }
    else:
        return {"type": "none"}


@app.get("/api/trajectory")
async def get_trajectory():
    """获取轨迹"""
    if unified_system:
        traj = unified_system.result.local_map.pose
        return {
            "trajectory": [
                {"x": p.x, "y": p.y, "theta": p.theta}
                for p in unified_system.slam.state.trajectory[-100:]  # 最近100个点
            ]
        }
    else:
        return {"trajectory": []}


# ============= 启动 =============

def main():
    print("="*60)
    print("   AGV Perception + SLAM Server v2.0")
    print("="*60)
    print("    Address: http://0.0.0.0:8080")
    print()
    print("    Endpoints:")
    print("      POST /api/detect     - 目标检测")
    print("      POST /api/fusion     - 融合感知 (视觉+激光)")
    print("      POST /api/navigate   - 导航决策")
    print("      GET  /api/pose       - 位姿估计")
    print("      GET  /api/map        - 获取地图")
    print()
    print("    C++ Integration Example:")
    print("    ```cpp")
    print("    // 发送图像+激光融合请求")
    print("    curl -X POST http://localhost:8080/api/fusion \\")
    print("      -H 'Content-Type: application/json' \\")
    print("      -d '{\"image_base64\": \"...\", \"lidar_scan\": [...]}'")
    print("    ```")
    print("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
