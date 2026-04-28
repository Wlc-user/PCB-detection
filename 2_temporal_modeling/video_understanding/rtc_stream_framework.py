"""
RTC 实时通信流处理框架
支持: WebRTC / 声网Agora / 腾讯云TRTC / ZEGO

架构:
    客户端 ←→ 信令服务器 ←→ 流媒体服务器 ←→ 视觉理解Pipeline

使用方式:
    python rtc_stream_framework.py --mode server --backend webrtc
    python rtc_stream_framework.py --mode server --backend agora
    python rtc_stream_framework.py --mode client --backend webrtc
"""

import os
import sys
import time
import asyncio
import logging
import argparse
import json
import base64
import queue
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# ============ 日志配置 ============
def setup_logger(name='RTCStream'):
    log_dir = Path('logs/rtc')
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'rtc_{timestamp}.log'
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fh = logging.FileHandler(log_file, encoding='utf-8')
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, log_file

logger, log_file = setup_logger()


# ============ RTC 后端类型 ============
class RTCBackend(Enum):
    WEBRTC = "webrtc"       # 原生WebRTC
    AGORA = "agora"         # 声网
    TRTC = "trtc"           # 腾讯云实时音视频
    ZEGO = "zego"           # ZEGO即构科技


# ============ 数据结构 ============
@dataclass
class StreamFrame:
    """流帧数据"""
    frame_id: str
    timestamp: float
    data: bytes  # 原始图像数据
    width: int
    height: int
    peer_id: str = ""  # 来源peer


@dataclass
class RTCConfig:
    """RTC配置"""
    backend: str = "webrtc"
    room_id: str = "default"
    user_id: str = ""
    
    # WebRTC配置
    stun_server: str = "stun:stun.l.google.com:19302"
    turn_server: str = ""
    
    # 声网配置
    agora_app_id: str = ""
    agora_token: str = ""
    agora_channel: str = ""
    
    # 腾讯云配置
    trtc_app_id: str = ""
    trtc_user_id: str = ""
    trtc_user_sig: str = ""
    trtc_room_id: int = 0
    
    # 处理配置
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_fps: int = 30


# ============ 视觉理解 Pipeline ============
class VisionPipeline:
    """视觉理解Pipeline"""
    
    def __init__(self, config: RTCConfig):
        self.config = config
        self.yolo_model = None
        self._load_models()
    
    def _load_models(self):
        """加载模型"""
        try:
            from ultralytics import YOLO
            model_path = 'models/yolov8/train/weights/best.pt'
            if Path(model_path).exists():
                self.yolo_model = YOLO(model_path)
                logger.info(f"✓ YOLO模型加载: {model_path}")
            else:
                logger.warning(f"⚠ 模型不存在: {model_path}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
    
    async def process(self, frame: StreamFrame) -> Dict:
        """处理帧"""
        start_time = time.time()
        
        # 解码图像
        import cv2
        np_arr = np.frombuffer(frame.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {'error': 'decode_failed', 'frame_id': frame.frame_id}
        
        detections = []
        
        # YOLO检测
        if self.yolo_model is not None:
            results = self.yolo_model(
                image,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                verbose=False,
            )
            
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    class_names = {
                        0: 'missing_hole', 1: 'mouse_bite', 2: 'open_circuit',
                        3: 'short', 4: 'spur', 5: 'spurious_copper', 6: 'normal'
                    }
                    
                    detections.append({
                        'class_name': class_names.get(cls_id, 'unknown'),
                        'confidence': round(conf, 4),
                        'bbox': [int(x) for x in [x1, y1, x2, y2]],
                    })
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            'frame_id': frame.frame_id,
            'timestamp': frame.timestamp,
            'detections': detections,
            'detection_count': len(detections),
            'processing_time_ms': round(processing_time, 2),
            'from_peer': frame.peer_id,
        }


# ============ WebRTC 实现 ============
class WebRTCRoom:
    """WebRTC房间管理"""
    
    def __init__(self, room_id: str):
        self.room_id = room_id
        self.peers: Dict[str, 'WebRTCPeer'] = {}
        self.pipeline = None
        self.message_queue = queue.Queue()
        logger.info(f"创建房间: {room_id}")
    
    def add_peer(self, peer_id: str, peer: 'WebRTCPeer'):
        self.peers[peer_id] = peer
        logger.info(f"Peer加入: {peer_id} (房间: {self.room_id})")
    
    def remove_peer(self, peer_id: str):
        if peer_id in self.peers:
            del self.peers[peer_id]
            logger.info(f"Peer离开: {peer_id}")
    
    async def broadcast(self, message: Dict, exclude_peer: str = None):
        """广播消息给房间内所有peer"""
        for peer_id, peer in self.peers.items():
            if peer_id != exclude_peer:
                await peer.send(message)


class WebRTCPeer:
    """WebRTC对等端"""
    
    def __init__(self, peer_id: str, room: WebRTCRoom):
        self.peer_id = peer_id
        self.room = room
        self.pc = None  # RTCPeerConnection
        self.data_channel = None
        self._message_callback: Optional[Callable] = None
    
    def set_message_callback(self, callback: Callable):
        self._message_callback = callback
    
    async def send(self, message: Dict):
        """发送消息"""
        if self.data_channel and self.data_channel.readyState == 'open':
            self.data_channel.send(json.dumps(message))
    
    async def handle_offer(self, offer: Dict) -> Dict:
        """处理offer并返回answer"""
        # 这里简化处理，实际需要用aiortc或类似库
        logger.info(f"[{self.peer_id}] 收到offer")
        return {'type': 'answer', 'sdp': 'v=0\r\n...'}
    
    async def handle_ice_candidate(self, candidate: Dict):
        """处理ICE候选"""
        logger.debug(f"[{self.peer_id}] ICE候选: {candidate.get('candidate', '')[:50]}...")
    
    async def handle_data_message(self, message: bytes):
        """处理数据通道消息"""
        try:
            data = json.loads(message.decode())
            
            if data.get('type') == 'frame':
                # 处理图像帧
                import cv2
                import numpy as np
                
                img_data = base64.b64decode(data.get('data', ''))
                frame = StreamFrame(
                    frame_id=data.get('frame_id', f'frame_{time.time()}'),
                    timestamp=data.get('timestamp', time.time()),
                    data=img_data,
                    width=data.get('width', 0),
                    height=data.get('height', 0),
                    peer_id=self.peer_id,
                )
                
                if self._message_callback:
                    result = await self._message_callback(frame)
                    await self.send(result)
            
            elif data.get('type') == 'result_request':
                # 结果请求
                pass
        
        except Exception as e:
            logger.error(f"消息处理错误: {e}")


class WebRTCSignalingServer:
    """WebRTC信令服务器"""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8080):
        self.host = host
        self.port = port
        self.rooms: Dict[str, WebRTCRoom] = {}
        self.peers: Dict[str, WebRTCPeer] = {}
        self.pipeline = None
        self._server = None
        logger.info(f"WebRTC信令服务器初始化: {host}:{port}")
    
    def set_pipeline(self, pipeline: VisionPipeline):
        self.pipeline = pipeline
    
    async def start(self):
        """启动服务器（简化版，使用aiohttp）"""
        try:
            from aiohttp import web
            
            async def websocket_handler(request):
                ws = web.WebSocketResponse()
                await ws.prepare(request)
                
                peer_id = request.query.get('peer_id', f'peer_{time.time()}')
                room_id = request.query.get('room_id', 'default')
                
                # 创建/加入房间
                if room_id not in self.rooms:
                    self.rooms[room_id] = WebRTCRoom(room_id)
                    if self.pipeline:
                        self.rooms[room_id].pipeline = self.pipeline
                
                room = self.rooms[room_id]
                peer = WebRTCPeer(peer_id, room)
                room.add_peer(peer_id, peer)
                self.peers[peer_id] = peer
                
                # 设置消息回调
                if self.pipeline:
                    async def on_frame(frame: StreamFrame):
                        return await self.pipeline.process(frame)
                    peer.set_message_callback(on_frame)
                
                logger.info(f"[{peer_id}] WebSocket连接建立")
                
                try:
                    async for msg in ws:
                        if msg.type == web.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            await self._handle_message(peer, room, data)
                        elif msg.type == web.WSMsgType.BINARY:
                            await peer.handle_data_message(msg.data)
                
                finally:
                    room.remove_peer(peer_id)
                    del self.peers[peer_id]
                    if not room.peers:
                        del self.rooms[room_id]
                
                return ws
            
            async def _handle_message(self, peer: WebRTCPeer, room: WebRTCRoom, data: Dict):
                msg_type = data.get('type')
                
                if msg_type == 'offer':
                    answer = await peer.handle_offer(data)
                    await peer.send(answer)
                    await room.broadcast({'type': 'peer_joined', 'peer_id': peer.peer_id}, exclude_peer=peer.peer_id)
                
                elif msg_type == 'answer':
                    logger.info(f"[{peer.peer_id}] 收到answer")
                
                elif msg_type == 'ice_candidate':
                    await peer.handle_ice_candidate(data)
                
                elif msg_type == 'broadcast_frame':
                    # 广播帧给房间内其他人
                    await room.broadcast(data, exclude_peer=peer.peer_id)
                
                elif msg_type == 'get_stats':
                    stats = {'peers': len(room.peers), 'room': room.room_id}
                    await peer.send({'type': 'stats', **stats})
            
            app = web.Application()
            app.router.add_get('/ws', websocket_handler)
            app.router.add_get('/health', lambda r: web.json_response({'status': 'ok'}))
            
            runner = web.AppRunner(app)
            await runner.setup()
            self._server = web.TCPSite(runner, self.host, self.port)
            await self._server.start()
            
            logger.info("=" * 50)
            logger.info(f"WebRTC信令服务器启动: ws://{self.host}:{self.port}")
            logger.info(f"连接示例: ws://localhost:{self.port}/ws?room_id=test&peer_id=user1")
            logger.info("=" * 50)
            
            # 永久运行
            await asyncio.Future()
        
        except ImportError:
            logger.error("需要安装aiohttp: pip install aiohttp")
            raise


# ============ 声网 Agora 实现 ============
class AgoraRTC:
    """
    声网 RTC SDK 封装
    需要: pip install agora-python-sdk
    """
    
    def __init__(self, config: RTCConfig):
        self.config = config
        self.client = None
        self.pipeline = None
        self._setup_client()
    
    def _setup_client(self):
        """设置声网客户端"""
        if not self.config.agora_app_id:
            logger.warning("⚠ 未配置声网 AppID，请设置 agora_app_id")
            return
        
        try:
            # 声网SDK (示例代码，实际需要安装agora包)
            logger.info(f"声网RTC初始化")
            logger.info(f"  AppID: {self.config.agora_app_id[:10]}...")
            logger.info(f"  Channel: {self.config.agora_channel}")
            # self.client = AgoraRtcEngine.create(...)
        except ImportError:
            logger.error("请安装声网SDK: pip install agora-python-sdk")
        except Exception as e:
            logger.error(f"声网初始化失败: {e}")
    
    def set_pipeline(self, pipeline: VisionPipeline):
        self.pipeline = pipeline
    
    def start(self):
        """启动"""
        if not self.client:
            logger.error("声网客户端未初始化")
            return
        
        logger.info("声网RTC启动")
        # self.client.join(channel, token, uid)
        # self.client.setVideoFrameCallback(...)
    
    def stop(self):
        """停止"""
        if self.client:
            logger.info("声网RTC停止")
            # self.client.leave()
    
    def send_frame(self, frame: bytes, width: int, height: int):
        """发送视频帧"""
        # 将帧数据发送出去
        pass
    
    def on_video_frame(self, frame_info: Dict):
        """视频帧回调"""
        if self.pipeline:
            frame = StreamFrame(
                frame_id=f"agora_{time.time()}",
                timestamp=time.time(),
                data=frame_info['data'],
                width=frame_info['width'],
                height=frame_info['height'],
            )
            return self.pipeline.process(frame)
    
    def broadcast_result(self, result: Dict):
        """广播结果"""
        # 发送检测结果
        pass


# ============ 腾讯云 TRTC 实现 ============
class TRTCRTC:
    """
    腾讯云实时音视频 SDK 封装
    需要: pip install tencentcloud-sdk-python-live
    """
    
    def __init__(self, config: RTCConfig):
        self.config = config
        self.client = None
        self.pipeline = None
        self._setup_client()
    
    def _setup_client(self):
        """设置TRTC客户端"""
        if not self.config.trtc_app_id:
            logger.warning("⚠ 未配置腾讯云 TRTC AppID")
            return
        
        try:
            logger.info(f"腾讯云TRTC初始化")
            logger.info(f"  AppID: {self.config.trtc_app_id}")
            logger.info(f"  RoomID: {self.config.trtc_room_id}")
        except Exception as e:
            logger.error(f"TRTC初始化失败: {e}")
    
    def set_pipeline(self, pipeline: VisionPipeline):
        self.pipeline = pipeline
    
    def start(self):
        """启动"""
        logger.info("腾讯云TRTC启动")
    
    def stop(self):
        """停止"""
        logger.info("腾讯云TRTC停止")


# ============ RTC 工厂 ============
class RTCFactory:
    """RTC后端工厂"""
    
    @staticmethod
    def create(backend: str, config: RTCConfig) -> 'BaseRTC':
        """创建RTC实例"""
        if backend == 'webrtc':
            return WebRTCRTC(config)
        elif backend == 'agora':
            return AgoraRTC(config)
        elif backend == 'trtc':
            return TRTCRTC(config)
        else:
            raise ValueError(f"不支持的后端: {backend}")


# WebRTC实际实现
class WebRTCRTC:
    """WebRTC RTC实现"""
    
    def __init__(self, config: RTCConfig):
        self.config = config
        self.server = None
        self.pipeline = None
    
    def set_pipeline(self, pipeline: VisionPipeline):
        self.pipeline = pipeline
    
    async def start_server(self, host='0.0.0.0', port=8080):
        """启动WebRTC信令服务器"""
        self.server = WebRTCSignalingServer(host, port)
        if self.pipeline:
            self.server.set_pipeline(self.pipeline)
        await self.server.start()
    
    async def start_client(self, server_url: str, room_id: str, user_id: str):
        """启动WebRTC客户端"""
        try:
            import websockets
            
            uri = f"{server_url}?room_id={room_id}&peer_id={user_id}"
            logger.info(f"连接WebRTC服务器: {uri}")
            
            async with websockets.connect(uri) as ws:
                logger.info("已连接，等待房间内其他人...")
                
                # 发送offer
                await ws.send(json.dumps({
                    'type': 'offer',
                    'sdp': 'v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\n...'
                }))
                
                # 接收消息
                async for msg in ws:
                    data = json.loads(msg)
                    if data.get('type') == 'answer':
                        logger.info("收到answer")
                    elif data.get('type') == 'peer_joined':
                        logger.info(f"用户加入: {data.get('peer_id')}")
                    elif data.get('type') == 'frame':
                        # 处理接收到的帧
                        pass
        
        except ImportError:
            logger.error("需要安装websockets: pip install websockets")


# ============ HTML客户端生成 ============
def generate_webrtc_html(output_path: str = 'webrtc_client.html'):
    """生成WebRTC测试HTML"""
    html = '''
<!DOCTYPE html>
<html>
<head>
    <title>RTC 视觉理解客户端</title>
    <style>
        body { font-family: Arial; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .video-container { display: flex; gap: 20px; flex-wrap: wrap; }
        video { width: 400px; background: #000; }
        .controls { margin: 20px 0; }
        button { padding: 10px 20px; margin: 5px; cursor: pointer; }
        #stats { background: #f5f5f5; padding: 10px; margin: 10px 0; }
        #log { background: #eee; padding: 10px; max-height: 200px; overflow-y: auto; font-family: monospace; }
        pre { white-space: pre-wrap; word-wrap: break-word; }
    </style>
</head>
<body>
    <h1>RTC 实时视觉理解</h1>
    
    <div class="controls">
        <label>房间号: <input type="text" id="roomId" value="test"></label>
        <label>用户ID: <input type="text" id="userId" value="user1"></label>
        <button onclick="joinRoom()">加入房间</button>
        <button onclick="leaveRoom()">离开房间</button>
        <button onclick="toggleCamera()">切换摄像头</button>
    </div>
    
    <div class="video-container">
        <div>
            <h3>本地视频</h3>
            <video id="localVideo" autoplay muted></video>
        </div>
        <div>
            <h3>处理结果</h3>
            <canvas id="resultCanvas"></canvas>
        </div>
    </div>
    
    <div id="stats">
        <strong>统计:</strong>
        <span id="fps">FPS: 0</span> |
        <span id="detections">检测: 0</span> |
        <span id="latency">延迟: 0ms</span>
    </div>
    
    <div id="log"></div>
    
    <script>
        let ws = null;
        let pc = null;
        let localStream = null;
        let sending = false;
        let frameCount = 0;
        let lastTime = performance.now();
        
        function log(msg) {
            document.getElementById('log').innerHTML += '<pre>' + msg + '</pre>';
        }
        
        async function joinRoom() {
            const roomId = document.getElementById('roomId').value;
            const userId = document.getElementById('userId').value;
            const serverUrl = 'ws://localhost:8080/ws';
            
            // 连接信令服务器
            ws = new WebSocket(serverUrl + '?room_id=' + roomId + '&peer_id=' + userId);
            
            ws.onopen = () => {
                log('[WS] 已连接');
                initLocalStream();
            };
            
            ws.onmessage = async (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'connected') {
                    log('[WS] 加入房间成功');
                } else if (data.type === 'result') {
                    updateStats(data);
                } else if (data.type === 'offer') {
                    await handleOffer(data);
                } else if (data.type === 'answer') {
                    await pc.setRemoteDescription(new RTCSessionDescription(data));
                } else if (data.type === 'ice_candidate') {
                    await pc.addIceCandidate(new RTCIceCandidate(data.candidate));
                }
            };
            
            ws.onclose = () => log('[WS] 连接关闭');
            ws.onerror = (e) => log('[WS] 错误: ' + e);
        }
        
        async function initLocalStream() {
            try {
                localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                document.getElementById('localVideo').srcObject = localStream;
                
                // 创建RTCPeerConnection
                pc = new RTCPeerConnection({
                    iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
                });
                
                pc.onicecandidate = (e) => {
                    if (e.candidate && ws) {
                        ws.send(JSON.stringify({
                            type: 'ice_candidate',
                            candidate: e.candidate
                        }));
                    }
                };
                
                // 添加本地视频轨道
                localStream.getTracks().forEach(track => pc.addTrack(track, localStream));
                
                // 创建数据通道
                const dc = pc.createDataChannel('stream');
                setupDataChannel(dc);
                
                // 创建并发送offer
                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);
                ws.send(JSON.stringify({ type: 'offer', sdp: offer.sdp }));
                
                log('[WebRTC] 开始推流');
                sending = true;
                sendFrames();
                
            } catch (e) {
                log('[错误] 无法获取摄像头: ' + e);
            }
        }
        
        function setupDataChannel(dc) {
            dc.onopen = () => log('[DC] 数据通道打开');
            dc.onmessage = (e) => {
                try {
                    const data = JSON.parse(e.data);
                    if (data.detections) {
                        drawResults(data);
                    }
                } catch (err) {}
            };
        }
        
        async function sendFrames() {
            if (!sending || !localStream) return;
            
            const video = document.getElementById('localVideo');
            const canvas = document.createElement('canvas');
            canvas.width = 640;
            canvas.height = 480;
            const ctx = canvas.getContext('2d');
            
            async function captureAndSend() {
                if (!sending) return;
                
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const blob = await new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.7));
                const reader = new FileReader();
                
                reader.onload = () => {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            type: 'frame',
                            frame_id: 'frame_' + frameCount++,
                            timestamp: Date.now(),
                            data: reader.result.split(',')[1]
                        }));
                    }
                };
                reader.readAsDataURL(blob);
                
                frameCount++;
                document.getElementById('fps').textContent = 'FPS: ' + (frameCount / ((performance.now() - lastTime) / 1000)).toFixed(1);
                
                setTimeout(captureAndSend, 100); // 10 FPS
            }
            
            captureAndSend();
        }
        
        function drawResults(data) {
            const canvas = document.getElementById('resultCanvas');
            const ctx = canvas.getContext('2d');
            canvas.width = 640;
            canvas.height = 480;
            ctx.drawImage(document.getElementById('localVideo'), 0, 0);
            
            ctx.strokeStyle = '#00ff00';
            ctx.fillStyle = '#00ff00';
            ctx.font = '16px Arial';
            
            data.detections.forEach(d => {
                const [x1, y1, x2, y2] = d.bbox;
                ctx.strokeRect(x1, y1, x2-x1, y2-y1);
                ctx.fillText(d.class_name + ' ' + d.confidence.toFixed(2), x1, y1-5);
            });
            
            document.getElementById('detections').textContent = '检测: ' + data.detection_count;
            document.getElementById('latency').textContent = '延迟: ' + data.processing_time_ms + 'ms';
        }
        
        function updateStats(data) {
            drawResults(data);
        }
        
        function toggleCamera() {
            sending = !sending;
            log(sending ? '[摄像头] 开启' : '[摄像头] 关闭');
        }
        
        function leaveRoom() {
            sending = false;
            if (ws) ws.close();
            if (pc) pc.close();
            if (localStream) localStream.getTracks().forEach(t => t.stop());
            log('[房间] 已离开');
        }
    </script>
</body>
</html>
'''
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"WebRTC客户端HTML已生成: {output_path}")


# ============ 主函数 ============
def main():
    parser = argparse.ArgumentParser(description='RTC 实时通信流处理')
    
    parser.add_argument('--mode', type=str, default='server',
                       choices=['server', 'client', 'generate-html'],
                       help='运行模式')
    
    # 后端选择
    parser.add_argument('--backend', type=str, default='webrtc',
                       choices=['webrtc', 'agora', 'trtc'],
                       help='RTC后端')
    
    # 服务器配置
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    
    # 房间配置
    parser.add_argument('--room-id', type=str, default='test')
    parser.add_argument('--user-id', type=str, default='user1')
    parser.add_argument('--server-url', type=str, default='ws://localhost:8080')
    
    # 声网配置
    parser.add_argument('--agora-app-id', type=str, default='')
    parser.add_argument('--agora-token', type=str, default='')
    parser.add_argument('--agora-channel', type=str, default='')
    
    # 腾讯云配置
    parser.add_argument('--trtc-app-id', type=str, default='')
    parser.add_argument('--trtc-user-id', type=str, default='')
    parser.add_argument('--trtc-user-sig', type=str, default='')
    parser.add_argument('--trtc-room-id', type=int, default=0)
    
    # 处理配置
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--fps', type=int, default=10)
    
    args = parser.parse_args()
    
    # 创建配置
    config = RTCConfig(
        backend=args.backend,
        room_id=args.room_id,
        user_id=args.user_id,
        agora_app_id=args.agora_app_id,
        agora_token=args.agora_token,
        agora_channel=args.agora_channel,
        trtc_app_id=args.trtc_app_id,
        trtc_user_id=args.trtc_user_id,
        trtc_user_sig=args.trtc_user_sig,
        trtc_room_id=args.trtc_room_id,
        conf_threshold=args.conf,
        max_fps=args.fps,
    )
    
    if args.mode == 'generate-html':
        generate_webrtc_html()
        return
    
    elif args.mode == 'server':
        logger.info("=" * 50)
        logger.info(f"RTC 服务器启动 ({args.backend})")
        logger.info("=" * 50)
        
        # 创建视觉理解Pipeline
        pipeline = VisionPipeline(config)
        
        # 创建RTC实例
        rtc = RTCFactory.create(args.backend, config)
        rtc.set_pipeline(pipeline)
        
        if args.backend == 'webrtc':
            asyncio.run(rtc.start_server(args.host, args.port))
        else:
            rtc.start()
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                rtc.stop()
    
    elif args.mode == 'client':
        logger.info("=" * 50)
        logger.info("RTC 客户端启动")
        logger.info("=" * 50)
        
        rtc = WebRTCRTC(config)
        
        if args.backend == 'webrtc':
            asyncio.run(rtc.start_client(args.server_url, args.room_id, args.user_id))
    
    logger.info(f"\n日志文件: {log_file}")


if __name__ == '__main__':
    main()
