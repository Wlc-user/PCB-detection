"""
简化版测试服务器 - WebSocket + HTTP
"""
import asyncio
import base64
import json
import logging
import sys
from pathlib import Path

import aiohttp
from aiohttp import web
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 全局模型
model = None

def load_model():
    global model
    model_path = 'models/yolov8/train/weights/best.pt'
    if Path(model_path).exists():
        logger.info(f"加载模型: {model_path}")
        model = YOLO(model_path)
        logger.info("模型加载成功!")
    else:
        logger.warning(f"模型不存在: {model_path}, 使用模拟模式")

async def detect_objects(image_bytes: bytes, conf: float = 0.25) -> dict:
    """检测图像中的对象"""
    if model is None:
        # 模拟模式
        return {
            "success": True,
            "detections": [
                {"class": "missing_hole", "confidence": 0.85, "x": 100, "y": 100, "w": 50, "h": 50}
            ],
            "processing_time_ms": 10
        }
    
    # 真实检测
    import numpy as np
    import cv2
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = model(img, conf=conf, verbose=False)
    
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            cls_name = model.names[cls]
            
            detections.append({
                "class": cls_name,
                "confidence": round(conf, 3),
                "x": int(x1), "y": int(y1),
                "w": int(x2-x1), "h": int(y2-y1)
            })
    
    return {
        "success": True,
        "detections": detections,
        "count": len(detections)
    }

# ============ WebSocket 处理 ============
async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    logger.info("WebSocket 客户端连接")
    
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    msg_type = data.get('type', '')
                    
                    if msg_type == 'image':
                        # 解码图像
                        img_data = base64.b64decode(data['data'])
                        conf = data.get('confidence', 0.25)
                        
                        # 检测
                        result = await detect_objects(img_data, conf)
                        
                        # 发送结果
                        await ws.send_json(result)
                        logger.info(f"检测完成: {result.get('count', 0)} 个目标")
                        
                    elif msg_type == 'ping':
                        await ws.send_json({"type": "pong"})
                        
                except Exception as e:
                    logger.error(f"处理消息错误: {e}")
                    await ws.send_json({"success": False, "error": str(e)})
                    
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"WebSocket 错误: {ws.exception()}")
                
    except Exception as e:
        logger.error(f"WebSocket 连接错误: {e}")
    
    logger.info("WebSocket 客户端断开")
    return ws

# ============ HTTP 处理 ============
async def http_detect(request):
    """HTTP 检测接口"""
    try:
        body = await request.json()
        img_data = base64.b64decode(body['data'])
        conf = body.get('confidence', 0.25)
        
        result = await detect_objects(img_data, conf)
        return web.json_response(result)
        
    except Exception as e:
        logger.error(f"HTTP 请求错误: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)

async def http_health(request):
    """健康检查"""
    return web.json_response({
        "status": "ok",
        "model_loaded": model is not None
    })

# ============ 主函数 ============
async def init_app():
    app = web.Application(client_max_size=10*1024*1024)  # 10MB
    
    # 路由
    app.router.add_get('/health', http_health)
    app.router.add_post('/detect', http_detect)
    app.router.add_get('/ws', websocket_handler)
    app.router.add_get('/ws/', websocket_handler)
    
    return app

def main():
    # 加载模型
    load_model()
    
    # 启动服务器
    app = init_app()
    
    logger.info("=" * 50)
    logger.info("启动测试服务器")
    logger.info("HTTP: http://localhost:8080/detect")
    logger.info("WebSocket: ws://localhost:8765/ws")
    logger.info("=" * 50)
    
    web.run_app(app, host='0.0.0.0', port=8080, 
                print=None, access_log=None)

if __name__ == '__main__':
    main()
