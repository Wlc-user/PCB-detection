"""
AGV Perception + SLAM + CQC 增强服务器
=======================================
整合视觉感知、SLAM定位、CQC质量增强的完整系统

新增CQC功能：
1. 图像质量预筛 - 过滤模糊/过暗/过亮图像
2. 对比学习增强 - 优化检测置信度
3. 课程学习调度 - 渐进式难度调整
"""

import cv2
import numpy as np
import base64
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

from logistics_perception import LogisticsPerception
from unified_perception_slam import (
    UnifiedPerceptionSLAM, SimpleAGVPerception,
    create_unified_system, FusionLevel,
    Pose2D, SLAMType, SLAMConfig
)
from cqc_algorithm import (
    CQCDetector, CQCConfig, ImageQualityAnalyzer,
    DifficultyLevel
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============== FastAPI应用 ==============
app = FastAPI(
    title="AGV Perception + SLAM + CQC",
    version="3.0.0",
    description="视觉感知 + SLAM定位 + CQC质量增强"
)

# ============== 全局系统 ==============
unified_system = None
simple_system = None
cqc_detector = None
quality_analyzer = None


def init_systems():
    """初始化所有系统"""
    global unified_system, simple_system, cqc_detector, quality_analyzer
    
    logger.info("=" * 50)
    logger.info("Initializing AGV + SLAM + CQC Systems...")
    logger.info("=" * 50)
    
    # 1. CQC系统 - 质量分析
    cqc_config = CQCConfig(
        initial_threshold=0.2,
        max_threshold=0.8,
        curriculum_steps=200,
        temperature=0.1,
        blur_threshold=80.0,
        brightness_min=30.0,
        brightness_max=220.0,
        base_confidence=0.25
    )
    cqc_detector = CQCDetector(cqc_config)
    quality_analyzer = ImageQualityAnalyzer()
    
    # 2. 加载YOLO模型到CQC
    model_path = "models/yolov8/train/weights/best.pt"
    import os
    if os.path.exists(model_path):
        try:
            from ultralytics import YOLO
            cqc_detector.load_yolo(model_path)
            logger.info(f"[CQC] YOLO model loaded: {model_path}")
        except Exception as e:
            logger.warning(f"[CQC] YOLO not available: {e}")
    
    logger.info("[CQC] Image Quality Analyzer ready")
    
    # 3. 简单感知系统
    try:
        simple_system = SimpleAGVPerception()
        logger.info("[OK] SimpleAGVPerception ready")
    except Exception as e:
        logger.error(f"[X] SimpleAGVPerception failed: {e}")
        simple_system = None
    
    # 4. 统一感知+SLAM系统
    try:
        unified_system = create_unified_system("fusion")
        logger.info("[OK] UnifiedPerceptionSLAM ready")
    except Exception as e:
        logger.error(f"[X] UnifiedPerceptionSLAM failed: {e}")
        unified_system = None
    
    logger.info("=" * 50)
    logger.info("All systems initialized!")
    logger.info("=" * 50)


def decode_image(b64_data: str) -> Optional[np.ndarray]:
    """解码base64图像"""
    try:
        img_bytes = base64.b64decode(b64_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


def process_with_cqc(frame: np.ndarray) -> Dict:
    """CQC增强处理流水线"""
    start_time = time.time()
    cqc_result = cqc_detector.detect(frame)
    quality = cqc_result['quality']
    
    return {
        "quality": {
            "quality_score": quality['quality_score'],
            "blur_score": quality['blur_score'],
            "brightness": quality['brightness'],
            "contrast": quality['contrast'],
            "noise": quality['noise'],
            "is_blurry": quality['is_blurry'],
            "is_usable": quality['usable'],
            "resolution": quality['resolution']
        },
        "difficulty": cqc_result['difficulty'],
        "curriculum_threshold": cqc_result['curriculum_threshold'],
        "detections": cqc_result['detections'],
        "enhanced_confidence": cqc_result['enhanced_confidence'],
        "processing_time_ms": (time.time() - start_time) * 1000,
        "cqc_stats": cqc_result['stats']
    }


# ============== 数据模型 ==============

class DetectionRequest(BaseModel):
    image_base64: str
    camera_id: str = "cam_0"
    use_cqc: bool = True


class FusionRequest(BaseModel):
    image_base64: str
    lidar_scan: Optional[List[float]] = None
    odometry: Optional[List[float]] = None
    camera_id: str = "cam_0"
    use_cqc: bool = True


class NavigationRequest(BaseModel):
    image_base64: str
    camera_id: str = "cam_0"
    target_distance: float = 2.0
    use_cqc: bool = True


# ============== API端点 ==============

@app.on_event("startup")
async def startup():
    init_systems()


@app.get("/")
async def root():
    return {
        "service": "AGV Perception + SLAM + CQC Server",
        "version": "3.0.0",
        "features": {
            "vision": "YOLO目标检测",
            "slam": "视觉/激光SLAM定位",
            "cqc": "课程学习+对比学习+质量评估"
        },
        "endpoints": {
            "POST /api/detect": "目标检测 (CQC增强)",
            "POST /api/fusion": "融合感知 (视觉+激光+CQC)",
            "POST /api/navigate": "导航决策",
            "GET /api/status": "系统状态",
            "GET /api/cqc-info": "CQC学习状态",
            "POST /api/quality-check": "仅质量检查",
            "GET /api/pose": "获取位姿",
            "GET /demo": "Web演示界面",
            "GET /health": "健康检查"
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "systems": {
            "cqc": cqc_detector is not None,
            "simple": simple_system is not None,
            "unified": unified_system is not None
        }
    }


@app.get("/api/status")
async def get_status():
    status = {
        "server_time": datetime.now().isoformat(),
        "cqc_enabled": cqc_detector is not None,
        "yolo_loaded": cqc_detector.yolo_model is not None if cqc_detector else False
    }
    if cqc_detector:
        status["cqc_info"] = cqc_detector.get_info()
    if simple_system:
        status["simple_system"] = {"running": True}
    if unified_system:
        status["unified_system"] = unified_system.get_status()
    return status


@app.get("/api/cqc-info")
async def cqc_info():
    if cqc_detector:
        return cqc_detector.get_info()
    return {"error": "CQC未初始化"}


@app.post("/api/quality-check")
async def quality_check(file: UploadFile = File(...)):
    """仅进行图像质量检查"""
    try:
        data = await file.read()
        frame = decode_image(base64.b64encode(data).decode())
        if frame is None:
            raise HTTPException(status_code=400, detail="无效图像")
        
        quality = quality_analyzer.analyze(frame)
        return {
            "quality_score": quality['quality_score'],
            "is_usable": quality['usable'],
            "details": {
                "blur_score": quality['blur_score'],
                "brightness": quality['brightness'],
                "contrast": quality['contrast'],
                "noise": quality['noise']
            },
            "recommendation": "use" if quality['usable'] else "resample"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect")
async def detect(request: DetectionRequest):
    """CQC增强目标检测"""
    import uuid
    request_id = str(uuid.uuid4())[:8]
    
    try:
        frame = decode_image(request.image_base64)
        if frame is None:
            raise HTTPException(status_code=400, detail="无效图像")
        
        result = {
            "request_id": request_id,
            "camera_id": request.camera_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # 感知检测
        if simple_system:
            perception = simple_system.update(frame)
            result["objects"] = [
                {
                    "class": obj.class_name,
                    "category": obj.category.value,
                    "bbox": list(obj.bbox),
                    "confidence": float(obj.confidence),
                    "track_id": obj.track_id
                }
                for obj in perception.objects
            ]
            result["scene_type"] = perception.scene_type.value
            result["path_clear"] = perception.path_clear
        
        # CQC增强
        if request.use_cqc and cqc_detector:
            cqc_result = process_with_cqc(frame)
            result.update(cqc_result)
        
        return result
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/fusion")
async def fusion(request: FusionRequest):
    """融合感知 + CQC增强"""
    import uuid
    request_id = str(uuid.uuid4())[:8]
    
    try:
        frame = decode_image(request.image_base64)
        if frame is None:
            raise HTTPException(status_code=400, detail="无效图像")
        
        result = {
            "request_id": request_id,
            "camera_id": request.camera_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # SLAM+感知融合
        if unified_system:
            scan = np.array(request.lidar_scan) if request.lidar_scan else None
            odometry = tuple(request.odometry) if request.odometry else None
            fusion_result = unified_system.update(frame, scan, odometry)
            
            result["objects"] = [
                {"class": obj.class_name, "category": obj.category.value,
                 "bbox": list(obj.bbox), "confidence": float(obj.confidence)}
                for obj in fusion_result.objects
            ]
            result["scene_type"] = fusion_result.scene_type.value
            result["pose"] = {
                "x": float(fusion_result.pose.x),
                "y": float(fusion_result.pose.y),
                "theta": float(fusion_result.pose.theta)
            }
            result["path_clear"] = fusion_result.path_clear
        
        # CQC增强
        if request.use_cqc and cqc_detector:
            cqc_result = process_with_cqc(frame)
            result.update(cqc_result)
        
        return result
    except Exception as e:
        logger.error(f"Fusion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/navigate")
async def navigate(request: NavigationRequest):
    """导航感知 + CQC质量检查"""
    try:
        frame = decode_image(request.image_base64)
        if frame is None:
            raise HTTPException(status_code=400, detail="无效图像")
        
        response = {
            "camera_id": request.camera_id,
            "timestamp": datetime.now().isoformat(),
            "target_distance": request.target_distance
        }
        
        # CQC质量检查
        if request.use_cqc and cqc_detector:
            quality = quality_analyzer.analyze(frame)
            response["image_quality"] = {
                "score": quality['quality_score'],
                "is_usable": quality['usable']
            }
            if not quality['usable']:
                response["can_proceed"] = False
                response["action"] = "skip_low_quality"
                return response
        
        # 感知检测
        if simple_system:
            perception = simple_system.update(frame)
            forward = [o for o in perception.obstacles if o.position[0] > 0]
            dangerous = [o for o in forward if o.position[0] < request.target_distance
                         and o.confidence > 0.7 and o.category in ["person", "vehicle", "forklift"]]
            
            response["obstacles"] = [{"x": o.position[0], "y": o.position[1], "category": o.category} for o in forward]
            response["can_proceed"] = len(dangerous) == 0
            response["action"] = "stop" if dangerous else "move_forward"
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pose")
async def get_pose():
    if unified_system:
        pose = unified_system.result.pose
        return {"x": float(pose.x), "y": float(pose.y), "theta": float(pose.theta), "timestamp": pose.timestamp}
    return {"error": "SLAM not available"}


# ============== Demo页面 ==============

@app.get("/demo", response_class=HTMLResponse)
async def demo():
    return HTMLResponse(content=get_demo_html())


def get_demo_html() -> str:
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AGV + SLAM + CQC Demo</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; color: #fff; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        h1 { text-align: center; padding: 25px 0; background: linear-gradient(90deg, #00d4ff, #7b2ff7, #f107a3); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.2em; }
        .subtitle { text-align: center; color: #888; margin-bottom: 25px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }
        .panel { background: rgba(255,255,255,0.05); border-radius: 16px; padding: 20px; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); }
        .panel h2 { color: #00d4ff; margin-bottom: 15px; font-size: 1.1em; }
        .upload-zone { border: 2px dashed #444; border-radius: 12px; padding: 30px; text-align: center; cursor: pointer; transition: all 0.3s; }
        .upload-zone:hover { border-color: #00d4ff; background: rgba(0,212,255,0.1); }
        .upload-zone input { display: none; }
        .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 15px; }
        .stat { background: rgba(0,0,0,0.3); padding: 12px; border-radius: 8px; text-align: center; }
        .stat-value { font-size: 1.5em; font-weight: bold; color: #00d4ff; }
        .stat-label { font-size: 0.8em; color: #888; }
        .quality-meter { height: 10px; background: #333; border-radius: 5px; margin: 10px 0; overflow: hidden; }
        .quality-fill { height: 100%; background: linear-gradient(90deg, #ff4444, #ffaa00, #00ff00); transition: width 0.3s; }
        .tag { display: inline-block; padding: 4px 12px; border-radius: 15px; font-size: 0.85em; font-weight: bold; }
        .tag-pass { background: #00ff00; color: #000; }
        .tag-fail { background: #ff4444; color: #fff; }
        .detection-item { background: rgba(255,100,100,0.2); padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; display: flex; justify-content: space-between; }
        #cameraVideo { width: 100%; border-radius: 8px; transform: scaleX(-1); }
        .btn { background: linear-gradient(90deg, #00d4ff, #7b2ff7); color: #fff; border: none; padding: 10px 25px; border-radius: 8px; cursor: pointer; margin-top: 10px; }
        .btn:hover { opacity: 0.9; }
        @media (max-width: 1024px) { .grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <h1>AGV Perception + SLAM + CQC</h1>
        <p class="subtitle">视觉感知 · 激光SLAM · CQC质量增强</p>
        
        <div class="grid">
            <div class="panel">
                <h2>上传图像</h2>
                <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
                    <p>点击上传图像</p>
                    <input type="file" id="fileInput" accept="image/*">
                </div>
                <img id="preview" style="max-width:100%;display:none;border-radius:8px;margin-top:10px">
                <label style="margin-top:10px;display:block"><input type="checkbox" id="useCqc" checked> 启用CQC增强</label>
                <button class="btn" onclick="detectImage()" style="width:100%">检测</button>
                
                <div style="margin-top:15px">
                    <button class="btn" onclick="toggleCamera()" style="width:100%">摄像头</button>
                    <video id="cameraVideo" autoplay playsinline style="display:none"></video>
                </div>
            </div>
            
            <div class="panel">
                <h2>CQC质量分析</h2>
                <div class="stats-grid">
                    <div class="stat"><div class="stat-value" id="qualityScore">-</div><div class="stat-label">质量分数</div></div>
                    <div class="stat"><div class="stat-value" id="blurScore">-</div><div class="stat-label">模糊度</div></div>
                    <div class="stat"><div class="stat-value" id="brightness">-</div><div class="stat-label">亮度</div></div>
                </div>
                <div class="quality-meter"><div class="quality-fill" id="qualityBar" style="width:0%"></div></div>
                <div id="qualityStatus"></div>
            </div>
            
            <div class="panel">
                <h2>检测结果</h2>
                <div id="resultArea"><p style="color:#666;text-align:center;padding:40px">上传图像后显示结果</p></div>
            </div>
        </div>
        
        <div class="grid" style="margin-top:20px">
            <div class="panel">
                <h2>CQC学习状态</h2>
                <div class="stats-grid">
                    <div class="stat"><div class="stat-value" id="curStep">0</div><div class="stat-label">课程步数</div></div>
                    <div class="stat"><div class="stat-value" id="threshold">0.5</div><div class="stat-label">阈值</div></div>
                    <div class="stat"><div class="stat-value" id="qualityRate">0%</div><div class="stat-label">质量帧率</div></div>
                </div>
            </div>
            <div class="panel">
                <h2>SLAM位姿</h2>
                <div class="stats-grid">
                    <div class="stat"><div class="stat-value" id="poseX">0.00</div><div class="stat-label">X (m)</div></div>
                    <div class="stat"><div class="stat-value" id="poseY">0.00</div><div class="stat-label">Y (m)</div></div>
                    <div class="stat"><div class="stat-value" id="poseTheta">0.00</div><div class="stat-label">θ (rad)</div></div>
                </div>
            </div>
            <div class="panel">
                <h2>导航状态</h2>
                <div id="navStatus"><p style="color:#666;text-align:center;padding:30px">等待检测...</p></div>
            </div>
        </div>
    </div>
    
    <canvas id="canvas" style="display:none"></canvas>
    
    <script>
        let cameraStream = null;
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        document.getElementById('fileInput').addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    document.getElementById('preview').src = e.target.result;
                    document.getElementById('preview').style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        });
        
        async function detectImage() {
            const file = document.getElementById('fileInput').files[0];
            if (!file) { alert('请选择图像'); return; }
            const formData = new FormData();
            formData.append('file', file);
            try {
                const resp = await fetch('/api/detect', { method: 'POST', body: formData });
                const result = await resp.json();
                showResults(result);
                updateCQCInfo();
            } catch (e) { alert('检测失败'); }
        }
        
        async function toggleCamera() {
            const video = document.getElementById('cameraVideo');
            if (cameraStream) {
                cameraStream.getTracks().forEach(t => t.stop());
                cameraStream = null;
                video.style.display = 'none';
                return;
            }
            try {
                cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: 640, height: 480 } });
                video.srcObject = cameraStream;
                video.style.display = 'block';
                setTimeout(captureFrame, 500);
            } catch (e) { alert('摄像头失败'); }
        }
        
        function captureFrame() {
            if (!cameraStream) return;
            const video = document.getElementById('cameraVideo');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);
            const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
            const base64 = dataUrl.split(',')[1];
            fetch('/api/detect', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image_base64: base64, use_cqc: true }) })
            .then(r => r.json()).then(result => { showResults(result); if (result.pose) updatePose(result.pose); });
            setTimeout(captureFrame, 500);
        }
        
        function showResults(result) {
            const q = result.quality || {};
            document.getElementById('qualityScore').textContent = (q.quality_score || 0).toFixed(2);
            document.getElementById('blurScore').textContent = (q.blur_score || 0).toFixed(0);
            document.getElementById('brightness').textContent = (q.brightness || 0).toFixed(0);
            document.getElementById('qualityBar').style.width = ((q.quality_score || 0) * 100) + '%';
            
            const isPass = (q.is_usable !== false) && (!result.detections || result.detections.length === 0);
            document.getElementById('qualityStatus').innerHTML = `<span class="tag ${q.is_usable !== false ? 'tag-pass' : 'tag-fail'}">${q.is_usable !== false ? '可用' : '不可用'}</span> <span class="tag tag-${result.difficulty || 'easy'}">${result.difficulty || 'easy'}</span>`;
            
            const detections = result.detections || [];
            let html = detections.length === 0 ? '<p style="color:#0f0;text-align:center;padding:30px">无缺陷</p>' : '<h3 style="color:#f66">发现缺陷:</h3>';
            detections.forEach(d => { html += `<div class="detection-item"><span>${d.label || 'defect'}</span><span style="color:#f66">${(d.enhanced_confidence||d.confidence||0).toFixed(2)}</span></div>`; });
            html += `<p style="margin-top:10px;color:#888">${(result.processing_time_ms||0).toFixed(1)} ms</p>`;
            document.getElementById('resultArea').innerHTML = html;
            document.getElementById('navStatus').innerHTML = `<p style="text-align:center"><span class="tag ${isPass ? 'tag-pass' : 'tag-fail'}">${isPass ? '可通行' : '需停障'}</span></p><p style="text-align:center;color:#888;margin-top:10px">${result.scene_type||'unknown'}</p>`;
        }
        
        function updatePose(pose) {
            document.getElementById('poseX').textContent = (pose.x||0).toFixed(2);
            document.getElementById('poseY').textContent = (pose.y||0).toFixed(2);
            document.getElementById('poseTheta').textContent = (pose.theta||0).toFixed(2);
        }
        
        function updateCQCInfo() {
            fetch('/api/cqc-info').then(r => r.json()).then(info => {
                document.getElementById('curStep').textContent = info.curriculum_step||0;
                document.getElementById('threshold').textContent = (info.curriculum_threshold||0.5).toFixed(2);
                document.getElementById('qualityRate').textContent = ((info.quality_rate||0)*100).toFixed(0)+'%';
            });
        }
        setInterval(updateCQCInfo, 2000);
        setTimeout(updateCQCInfo, 1000);
    </script>
</body>
</html>
"""


# ============== 启动 ==============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AGV + SLAM + CQC Server")
    parser.add_argument("--port", type=int, default=8080, help="服务端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务地址")
    args = parser.parse_args()
    
    print("=" * 60)
    print("   AGV Perception + SLAM + CQC Server v3.0")
    print("=" * 60)
    print(f"    Address: http://{args.host}:{args.port}")
    print()
    print("    Endpoints:")
    print("      POST /api/detect     - CQC增强检测")
    print("      POST /api/fusion     - 融合感知+CQC")
    print("      POST /api/navigate   - 导航决策")
    print("      GET  /api/cqc-info   - CQC学习状态")
    print("      GET  /demo          - Web演示界面")
    print("=" * 60)
    
    uvicorn.run(app, host=args.host, port=args.port)
