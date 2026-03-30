"""
FastAPI 工业缺陷检测服务
高性能REST API服务，支持多模型推理
"""

import os
import io
import base64
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import torch
import time
from contextlib import asynccontextmanager
import uvicorn
from dataclasses import dataclass
from enum import Enum
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 模型类型枚举
class ModelType(str, Enum):
    YOLOV8 = "yolov8"
    TENSORRT = "tensorrt"
    ONNX = "onnx"


# 请求模型
class DetectionRequest(BaseModel):
    """检测请求"""
    model_type: Optional[ModelType] = ModelType.YOLOV8
    conf_threshold: float = 0.15  # 默认降低以提高召回率
    iou_threshold: float = 0.45
    imgsz: int = 640


# 全局模型缓存
class ModelCache:
    """模型缓存管理器"""
    
    def __init__(self):
        self.models = {}
        self.model_info = {}
    
    def load_model(self, model_path: str, model_type: ModelType = ModelType.YOLOV8):
        """加载模型"""
        if model_path in self.models:
            logger.info(f"模型已加载: {model_path}")
            return self.models[model_path]
        
        logger.info(f"加载模型: {model_path} ({model_type})")
        
        try:
            if model_type == ModelType.YOLOV8:
                from ultralytics import YOLO
                model = YOLO(model_path)
            elif model_type == ModelType.ONNX:
                import onnxruntime as ort
                model = ort.InferenceSession(
                    model_path, 
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
            else:
                # TensorRT
                import tensorrt as trt
                model = self._load_tensorrt(model_path)
            
            self.models[model_path] = model
            self.model_info[model_path] = {
                'type': model_type,
                'loaded_at': time.time(),
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
            
            logger.info(f"模型加载成功: {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")
    
    def _load_tensorrt(self, engine_path: str):
        """加载TensorRT引擎"""
        # 简化实现，实际需要TensorRT runtime
        logger.warning("TensorRT加载需要安装tensorrt库")
        return None
    
    def get_model(self, model_path: str):
        """获取模型"""
        return self.models.get(model_path)


# 全局缓存
model_cache = ModelCache()

# 默认模型路径
DEFAULT_MODEL = "models/yolov8/train/weights/best.pt"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时预加载模型
    logger.info("正在初始化应用...")
    
    # 检查是否有训练好的模型
    if os.path.exists(DEFAULT_MODEL):
        try:
            model_cache.load_model(DEFAULT_MODEL, ModelType.YOLOV8)
            logger.info(f"默认模型已加载: {DEFAULT_MODEL}")
        except Exception as e:
            logger.warning(f"默认模型加载失败: {e}")
    else:
        logger.warning(f"默认模型不存在: {DEFAULT_MODEL}")
    
    logger.info("应用初始化完成")
    yield
    
    # 关闭时清理资源
    logger.info("正在清理资源...")
    model_cache.models.clear()


# 创建FastAPI应用
app = FastAPI(
    title="工业缺陷检测API",
    description="基于YOLOv8的高性能工业缺陷检测服务",
    version="2.0.0",
    lifespan=lifespan
)


def preprocess_image(image: bytes, target_size: int = 640) -> np.ndarray:
    """图像预处理"""
    # 解码图像
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("无法解码图像")
    
    return img


def postprocess_results(results, conf_threshold: float = 0.15) -> dict:
    """后处理检测结果"""
    detections = []
    
    if results is None:
        return {'detections': [], 'count': 0}
    
    # 提取检测结果
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        
        for box in boxes:
            # 获取边界框
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # 过滤低置信度
            if conf < conf_threshold:
                continue
            
            detections.append({
                'bbox': {
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2),
                    'width': float(x2 - x1),
                    'height': float(y2 - y1)
                },
                'confidence': conf,
                'class_id': cls,
                'class_name': result.names[cls]
            })
    
    return {
        'detections': detections,
        'count': len(detections)
    }


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "工业缺陷检测API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "detect": "/detect (POST)",
            "detect_base64": "/detect/base64 (POST)",
            "detect_url": "/detect/url (POST)",
            "batch": "/detect/batch (POST)",
            "model_info": "/model/info (GET)",
            "stats": "/stats (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }


@app.post("/detect")
async def detect_image(
    file: UploadFile = File(...),
    model_path: Optional[str] = None,
    conf_threshold: Optional[float] = 0.15,  # 降低以提高召回率
    iou_threshold: Optional[float] = 0.45,
    imgsz: Optional[int] = 640,
    return_image: Optional[bool] = False
):
    """
    图像检测接口
    
    Args:
        file: 上传的图像文件
        model_path: 模型路径
        conf_threshold: 置信度阈值
        iou_threshold: IOU阈值
        imgsz: 输入图像尺寸
        return_image: 是否返回标注图像
    """
    start_time = time.time()
    
    # 使用默认模型或指定模型
    model_path = model_path or DEFAULT_MODEL
    
    # 检查模型是否存在
    if model_path and not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"模型不存在: {model_path}")
    
    try:
        # 读取图像
        image = await file.read()
        img = preprocess_image(image)
        
        # 加载模型
        model = model_cache.load_model(model_path, ModelType.YOLOV8)
        
        # 推理
        results = model.predict(
            img,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            verbose=False
        )
        
        # 后处理
        response = postprocess_results(results, conf_threshold)
        response['inference_time'] = time.time() - start_time
        
        # 如果需要返回标注图像
        if return_image:
            annotated = results[0].plot()
            _, encoded = cv2.imencode('.jpg', annotated)
            response['annotated_image'] = base64.b64encode(encoded).decode('utf-8')
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"检测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/base64")
async def detect_base64(
    image_data: str,
    model_path: Optional[str] = None,
    conf_threshold: Optional[float] = 0.15,
    iou_threshold: Optional[float] = 0.45
):
    """Base64图像检测"""
    start_time = time.time()
    
    try:
        # 解码Base64
        image = base64.b64decode(image_data)
        img = preprocess_image(image)
        
        # 加载模型
        model_path = model_path or DEFAULT_MODEL
        model = model_cache.load_model(model_path, ModelType.YOLOV8)
        
        # 推理
        results = model.predict(
            img,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        response = postprocess_results(results, conf_threshold)
        response['inference_time'] = time.time() - start_time
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/batch")
async def detect_batch(
    files: List[UploadFile] = File(...),
    model_path: Optional[str] = None,
    conf_threshold: Optional[float] = 0.15,
    iou_threshold: Optional[float] = 0.45
):
    """批量检测"""
    start_time = time.time()
    
    model_path = model_path or DEFAULT_MODEL
    model = model_cache.load_model(model_path, ModelType.YOLOV8)
    
    results = []
    
    for file in files:
        try:
            image = await file.read()
            img = preprocess_image(image)
            
            # 推理
            result = model.predict(
                img,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            detections = postprocess_results(result, conf_threshold)
            detections['filename'] = file.filename
            results.append(detections)
            
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e),
                'detections': [],
                'count': 0
            })
    
    total_time = time.time() - start_time
    
    return {
        'total_images': len(files),
        'total_time': total_time,
        'avg_time': total_time / len(files) if files else 0,
        'results': results
    }


@app.get("/model/info")
async def model_info(model_path: Optional[str] = None):
    """获取模型信息"""
    model_path = model_path or DEFAULT_MODEL
    
    if model_path not in model_cache.model_info:
        if os.path.exists(model_path):
            model_cache.load_model(model_path, ModelType.YOLOV8)
        else:
            raise HTTPException(status_code=404, detail="模型未加载")
    
    info = model_cache.model_info[model_path].copy()
    info['model_path'] = model_path
    
    # 添加模型大小
    if os.path.exists(model_path):
        info['model_size_mb'] = os.path.getsize(model_path) / (1024 * 1024)
    
    return info


@app.get("/stats")
async def stats():
    """获取服务统计信息"""
    return {
        'loaded_models': len(model_cache.models),
        'models': list(model_cache.model_info.keys()),
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'torch_version': torch.__version__
    }


@app.post("/model/load")
async def load_model(model_path: str, model_type: ModelType = ModelType.YOLOV8):
    """手动加载模型"""
    try:
        model_cache.load_model(model_path, model_type)
        return {"status": "success", "model_path": model_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/unload")
async def unload_model(model_path: str):
    """卸载模型"""
    if model_path in model_cache.models:
        del model_cache.models[model_path]
        if model_path in model_cache.model_info:
            del model_cache.model_info[model_path]
        return {"status": "success", "message": f"模型已卸载: {model_path}"}
    else:
        raise HTTPException(status_code=404, detail="模型未加载")


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """启动服务"""
    uvicorn.run(
        "defect_detection_api:app",
        host=host,
        port=port,
        reload=reload,
        workers=1,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    
    print(f"启动服务: http://{args.host}:{args.port}")
    start_server(args.host, args.port, args.reload)
