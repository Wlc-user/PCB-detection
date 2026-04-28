"""
REST API 感知服务 - 更简单的C++集成方案
==========================================
无需gRPC，直接HTTP POST/GET
"""

import cv2
import numpy as np
import base64
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from logistics_perception import LogisticsPerception, ObjectCategory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AGV Perception API", version="1.0.0")

# 全局感知器
perception = None


def init_perception():
    global perception
    perception = LogisticsPerception()
    try:
        perception.load_models()
        logger.info("[OK] Perception models loaded")
    except Exception as e:
        logger.warning(f"[!] Model load failed: {e}")


# ============= 数据模型 =============

class DetectionRequest(BaseModel):
    image_base64: str
    camera_id: str = "cam_0"


class DetectionResponse(BaseModel):
    request_id: str
    timestamp: str
    scene_type: str
    objects: List[Dict]
    stats: Dict
    control_recommendation: Dict


class ControlCommand(BaseModel):
    command: str  # stop, move_forward, turn_left, turn_right, slow_down
    speed: float
    angle: float
    message: str


# ============= API 端点 =============

@app.post("/api/detect", response_model=DetectionResponse)
async def detect(request: DetectionRequest):
    """检测图像"""
    try:
        # 解码图像
        img_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # 感知
        result = perception.understand_scene(frame)
        
        # 生成控制建议
        control = make_control_recommendation(result)
        
        return DetectionResponse(
            request_id=f"{request.camera_id}_{int(result.timestamp * 1000)}",
            timestamp=datetime.fromtimestamp(result.timestamp).isoformat(),
            scene_type=result.scene_type.value,
            objects=[
                {
                    "track_id": obj.track_id,
                    "class_name": obj.class_name,
                    "category": obj.category.value,
                    "confidence": obj.confidence,
                    "bbox": list(obj.bbox),
                    "center": obj.center,
                }
                for obj in result.objects
            ],
            stats=result.stats,
            control_recommendation=control
        )
        
    except Exception as e:
        logger.error(f"[!] Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect/file")
async def detect_file(file: UploadFile = File(...)):
    """上传文件检测"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        result = perception.understand_scene(frame)
        control = make_control_recommendation(result)
        
        return {
            "request_id": f"file_{int(result.timestamp * 1000)}",
            "timestamp": datetime.fromtimestamp(result.timestamp).isoformat(),
            "scene_type": result.scene_type.value,
            "stats": result.stats,
            "control": control,
            "objects_count": len(result.objects)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def status():
    """服务状态"""
    return {
        "status": "running",
        "model_loaded": perception._detector is not None,
        "uptime": datetime.now().isoformat()
    }


def make_control_recommendation(result) -> Dict[str, Any]:
    """生成控制建议"""
    cmd = {
        "command": "move_forward",
        "speed": 0.5,
        "angle": 0.0,
        "message": "Path clear",
        "danger_level": 0.0
    }
    
    # 分析障碍物
    obstacles = result.get_obstacles()
    persons = result.get_persons()
    
    all_dangerous = obstacles + persons
    
    for obj in all_dangerous:
        conf = obj.confidence
        center_y = obj.center[1] / 720  # 归一化
        
        if conf > 0.7 and center_y > 0.7:
            cmd["command"] = "stop"
            cmd["speed"] = 0.0
            cmd["message"] = f"Emergency stop: {obj.class_name} too close"
            cmd["danger_level"] = 1.0
            break
        elif conf > 0.5 and center_y > 0.5:
            cmd["command"] = "slow_down"
            cmd["speed"] = 0.2
            cmd["message"] = f"Slow down: {obj.class_name} ahead"
            cmd["danger_level"] = 0.5
    
    return cmd


# ============= Web界面 =============

@app.get("/")
async def index():
    return {
        "service": "AGV Perception API",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/detect": "Detect image (base64)",
            "POST /api/detect/file": "Upload file to detect",
            "GET /api/status": "Service status"
        }
    }


@app.get("/demo")
async def demo():
    return FileResponse("templates/rest_demo.html")


# ============= 主程序 =============

def main():
    print("""
============================================================
    REST API Perception Server for AGV
============================================================
    Address: http://0.0.0.0:8080
    
    C++ Integration Example:
    ```cpp
    #include <curl/curl.h>
    
    std::string detect(cv::Mat& frame) {
        // 编码图像
        std::vector<unsigned char> buf;
        cv::imencode(".jpg", frame, buf);
        std::string img_base64 = base64_encode(buf.data(), buf.size());
        
        // 发送请求
        CURL* curl = curl_easy_init();
        curl_easy_setopt(curl, CURL_URL, "http://192.168.1.100:8080/api/detect");
        // ... set headers and POST data
    }
    ```
============================================================
    """)
    
    init_perception()
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
