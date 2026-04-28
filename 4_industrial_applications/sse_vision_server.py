"""
SSE Vision Server - 轻量级实时检测推送服务
适合浏览器实时展示检测结果

Author: Vision Team
"""

import asyncio
import base64
import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, Set, Optional, Callable, Any
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SSE Vision Server", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


class EventType(str, Enum):
    DETECTION = "detection"
    FRAME = "frame"
    CHAT = "chat"
    TTS = "tts"
    ALERT = "alert"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


@dataclass
class DetectionResult:
    frame_id: str
    timestamp: str
    detections: list
    fps: float
    latency_ms: float
    image_base64: Optional[str] = None


@dataclass
class ChatMessage:
    message_id: str
    role: str  # user / assistant / system
    content: str
    timestamp: str
    audio_url: Optional[str] = None


# ============= SSE 客户端管理 =============

class SSEClientManager:
    """SSE 客户端管理器"""
    
    def __init__(self):
        self.clients: Dict[str, Set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()
    
    async def subscribe(self, channel: str) -> asyncio.Queue:
        """订阅频道，返回消息队列"""
        queue = asyncio.Queue(maxsize=100)
        async with self._lock:
            if channel not in self.clients:
                self.clients[channel] = set()
            self.clients[channel].add(queue)
        logger.info(f"[SSE] Client subscribed to {channel}, total: {len(self.clients.get(channel, []))}")
        return queue
    
    async def unsubscribe(self, channel: str, queue: asyncio.Queue):
        """取消订阅"""
        async with self._lock:
            if channel in self.clients:
                self.clients[channel].discard(queue)
                if not self.clients[channel]:
                    del self.clients[channel]
        logger.info(f"[SSE] Client unsubscribed from {channel}")
    
    async def publish(self, channel: str, event_type: EventType, data: Any):
        """发布消息到频道"""
        if channel not in self.clients:
            return
        
        message = f"event: {event_type.value}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        
        # 复制队列列表避免迭代时修改
        clients_snapshot = list(self.clients[channel])
        for queue in clients_snapshot:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                logger.warning(f"[SSE] Queue full, dropping message")
    
    async def broadcast(self, event_type: EventType, data: Any):
        """广播到所有频道"""
        for channel in list(self.clients.keys()):
            await self.publish(channel, event_type, data)


# 全局客户端管理器
client_manager = SSEClientManager()

# SSE 事件生成器
async def sse_generator(channel: str, client_id: str):
    """SSE 事件流生成器"""
    queue = await client_manager.subscribe(channel)
    
    # 发送连接成功事件
    yield f"event: connected\ndata: {json.dumps({'client_id': client_id, 'channel': channel})}\n\n"
    
    try:
        while True:
            # 带超时的队列获取
            try:
                message = await asyncio.wait_for(queue.get(), timeout=30)
                yield message
            except asyncio.TimeoutError:
                # 发送心跳
                yield f"event: heartbeat\ndata: {json.dumps({'time': time.time()})}\n\n"
    except asyncio.CancelledError:
        pass
    finally:
        await client_manager.unsubscribe(channel, queue)


# ============= 业务逻辑 =============

class VisionService:
    """视觉检测服务"""
    
    def __init__(self):
        self.model = None
        self.llm_client = None
        self.tts_client = None
        self._initialized = False
    
    async def initialize(self):
        """初始化模型"""
        if self._initialized:
            return
        
        # 加载 YOLO 模型
        try:
            from ultralytics import YOLO
            model_path = "models/yolov8/train/weights/best.pt"
            if Path(model_path).exists():
                self.model = YOLO(model_path)
                logger.info("[OK] YOLO model loaded")
            else:
                logger.warning("[!] YOLO model not found, using demo mode")
        except ImportError:
            logger.warning("[!] Ultralytics not installed")
        
        self._initialized = True
    
    async def detect(self, frame: np.ndarray, conf: float = 0.25) -> list:
        """检测缺陷"""
        if self.model is None:
            # 模拟检测
            return [{"label": "demo_defect", "confidence": 0.75, "bbox": [100, 100, 200, 200]}]
        
        results = self.model(frame, conf=conf, verbose=False)
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.model.names.get(cls, f"class_{cls}")
                
                detections.append({
                    "label": label,
                    "confidence": round(conf, 2),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })
        
        return detections
    
    async def analyze_with_llm(self, image_base64: str, question: str = None) -> str:
        """用 LLM 分析图像"""
        if self.llm_client is None:
            # 模拟 LLM 分析
            return "检测到电路板图像，分析正常，未发现明显缺陷。"
        
        # 调用 DeepSeek Vision API
        # ...
        return "分析完成"


class ChatService:
    """对话服务"""
    
    def __init__(self, vision_service: VisionService):
        self.vision = vision_service
        self.conversation_history: Dict[str, list] = {}
    
    async def chat(self, client_id: str, message: str) -> str:
        """处理对话"""
        # 保存历史
        if client_id not in self.conversation_history:
            self.conversation_history[client_id] = []
        
        self.conversation_history[client_id].append({
            "role": "user",
            "content": message
        })
        
        # 调用 LLM
        try:
            from openai import AsyncOpenAI
            
            messages = [
                {"role": "system", "content": "你是一个专业的PCB缺陷检测助手，回答用户关于电路板缺陷检测的问题。"}
            ] + self.conversation_history[client_id][-10:]
            
            client = AsyncOpenAI(api_key="your-api-key", base_url="https://api.deepseek.com")
            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False
            )
            
            reply = response.choices[0].message.content
            
        except Exception as e:
            logger.warning(f"[LLM] Error: {e}")
            reply = f"收到消息: {message[:50]}... 我正在分析图像内容。"
        
        # 保存回复
        self.conversation_history[client_id].append({
            "role": "assistant",
            "content": reply
        })
        
        return reply
    
    async def text_to_speech(self, text: str, output_path: str = "output/sse_tts.mp3") -> Optional[str]:
        """TTS 合成"""
        try:
            from gtts import gTTS
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            tts = gTTS(text=text[:500], lang='zh')
            tts.save(output_path)
            
            return output_path
        except Exception as e:
            logger.warning(f"[TTS] Error: {e}")
            return None


# 全局服务
vision_service = VisionService()
chat_service = ChatService(vision_service)

# 全局配置
sse_config = {
    "channels": ["detections", "chat", "alerts"],
    "default_confidence": 0.25,
    "frame_skip": 2,  # 每隔 N 帧处理一次
}


# ============= API 端点 =============

@app.on_event("startup")
async def startup():
    """启动时初始化"""
    await vision_service.initialize()
    logger.info("[OK] SSE Vision Server started on http://0.0.0.0:8000")


@app.get("/")
async def root():
    """服务信息"""
    return {
        "name": "SSE Vision Server",
        "version": "2.0.0",
        "endpoints": {
            "sse_detections": "/sse/detections",
            "sse_chat": "/sse/chat",
            "sse_all": "/sse/all",
            "detect_image": "POST /api/detect",
            "chat": "POST /api/chat",
            "tts": "POST /api/tts",
            "upload": "POST /api/upload",
        }
    }


# ============= SSE 端点 =============

@app.get("/sse/{channel}")
async def sse_stream(channel: str, request: Request):
    """SSE 流端点"""
    if channel not in sse_config["channels"] and channel != "all":
        return JSONResponse({"error": "Invalid channel"}, status_code=400)
    
    client_id = str(uuid.uuid4())[:8]
    
    logger.info(f"[SSE] New client: {client_id} on channel: {channel}")
    
    # 如果是 all 频道，合并所有事件
    if channel == "all":
        async def merged_generator():
            detection_queue = await client_manager.subscribe("detections")
            chat_queue = await client_manager.subscribe("chat")
            alert_queue = await client_manager.subscribe("alerts")
            
            yield f"event: connected\ndata: {json.dumps({'client_id': client_id, 'channel': 'all'})}\n\n"
            
            try:
                while True:
                    # 同时监听多个队列
                    tasks = []
                    for q in [detection_queue, chat_queue, alert_queue]:
                        task = asyncio.create_task(q.get())
                        tasks.append((q, task))
                    
                    done, pending = await asyncio.wait(
                        [t for _, t in tasks],
                        timeout=30,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    for q, task in tasks:
                        if not task.done():
                            task.cancel()
                    
                    for task in done:
                        try:
                            msg = task.result()
                            yield msg
                        except asyncio.CancelledError:
                            pass
                    
                    # 如果超时，发送心跳
                    if not done:
                        yield f"event: heartbeat\ndata: {json.dumps({'time': time.time()})}\n\n"
                        
            except asyncio.CancelledError:
                pass
            finally:
                await client_manager.unsubscribe("detections", detection_queue)
                await client_manager.unsubscribe("chat", chat_queue)
                await client_manager.unsubscribe("alerts", alert_queue)
        
        return StreamingResponse(merged_generator(), media_type="text/event-stream")
    
    return StreamingResponse(
        sse_generator(channel, client_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ============= REST API 端点 =============

class DetectRequest(BaseModel):
    image: str  # base64 encoded
    conf: float = 0.25


class ChatRequest(BaseModel):
    message: str
    client_id: str = None


class TTSRequest(BaseModel):
    text: str
    lang: str = "zh"


@app.post("/api/detect")
async def detect_image(req: DetectRequest):
    """检测图像"""
    start_time = time.time()
    
    # 解码图像
    try:
        img_data = base64.b64decode(req.image)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return JSONResponse({"error": "Invalid image"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    
    # 检测
    detections = await vision_service.detect(frame, req.conf)
    
    # 绘制检测框
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{det['label']} {det['confidence']:.2f}", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 编码结果
    _, buffer = cv2.imencode('.jpg', frame)
    result_base64 = base64.b64encode(buffer).decode()
    
    latency = (time.time() - start_time) * 1000
    
    return {
        "detections": detections,
        "result_image": result_base64,
        "latency_ms": round(latency, 1),
        "count": len(detections)
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """对话接口"""
    client_id = req.client_id or str(uuid.uuid4())[:8]
    
    # 处理对话
    reply = await chat_service.chat(client_id, req.message)
    
    # 广播到 SSE 频道
    await client_manager.publish("chat", EventType.CHAT, {
        "client_id": client_id,
        "message": reply,
        "timestamp": datetime.now().isoformat()
    })
    
    return {
        "reply": reply,
        "client_id": client_id
    }


@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    """TTS 合成"""
    output_path = f"output/tts_{int(time.time())}.mp3"
    result = await chat_service.text_to_speech(req.text, output_path)
    
    if result:
        return {"audio_url": f"/audio/{Path(result).name}"}
    return JSONResponse({"error": "TTS failed"}, status_code=500)


@app.post("/api/upload")
async def upload_frame(request: Request):
    """接收图像帧并处理"""
    start_time = time.time()
    
    # 读取原始数据
    body = await request.body()
    
    # 尝试解析 JSON
    try:
        data = json.loads(body)
        if "image" in data:
            img_data = base64.b64decode(data["image"])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            return JSONResponse({"error": "No image data"}, status_code=400)
    except:
        # 直接作为图像数据
        nparr = np.frombuffer(body, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    
    # 检测
    conf = data.get("conf", 0.25) if 'data' in dir() else 0.25
    detections = await vision_service.detect(frame, conf)
    
    # 绘制
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det['label']} {det['confidence']:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 编码
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    result_base64 = base64.b64encode(buffer).decode()
    
    latency = (time.time() - start_time) * 1000
    frame_id = str(uuid.uuid4())[:8]
    
    # 广播到 SSE
    await client_manager.publish("detections", EventType.DETECTION, {
        "frame_id": frame_id,
        "timestamp": datetime.now().isoformat(),
        "detections": detections,
        "count": len(detections),
        "latency_ms": round(latency, 1),
        "image": result_base64[:5000] + "..." if len(result_base64) > 5000 else result_base64
    })
    
    return {
        "frame_id": frame_id,
        "detections": detections,
        "count": len(detections),
        "latency_ms": round(latency, 1)
    }


# ============= 静态文件 =============

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """获取音频文件"""
    from fastapi.responses import FileResponse
    
    audio_path = Path("output") / filename
    if audio_path.exists():
        return FileResponse(audio_path, media_type="audio/mpeg")
    return JSONResponse({"error": "Not found"}, status_code=404)


# ============= 主程序 =============

def main():
    """启动服务"""
    print("""
===============================================
       SSE Vision Server v2.0.0
===============================================

SSE Endpoints:
  GET  /sse/detections  - Detection stream
  GET  /se/chat         - Chat stream
  GET  /se/all          - All events merged

REST API:
  POST /api/detect      - Detect image
  POST /api/chat        - Chat
  POST /api/tts         - Text-to-Speech
  POST /api/upload      - Upload frame

Browser Demo:
  http://localhost:8000/templates/sse_demo.html
===============================================
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
