"""
gRPC + WebSocket 融合架构
结合两者优势:
- gRPC: 服务间高性能通信
- WebSocket: 前端实时交互

架构:
    [浏览器/客户端] ←WebSocket→ [网关服务] ←gRPC→ [视觉处理服务]
                                          ↓
                                    [YOLO检测模型]

使用方式:
    # 1. 启动网关服务 (WebSocket + gRPC Client)
    python grpc_websocket_server.py --mode gateway --ws-port 8765 --grpc-port 50051

    # 2. 启动视觉处理服务 (gRPC Server)
    python grpc_websocket_server.py --mode processor --grpc-port 50051

    # 3. 测试客户端
    python grpc_websocket_client.py --image test.jpg
"""

import os
import sys
import time
import asyncio
import logging
import argparse
import json
import base64
import threading
import queue
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import grpc
from concurrent import futures
import numpy as np
import cv2

# gRPC 生成代码
sys.path.insert(0, str(Path(__file__).parent))

# ============ 日志配置 ============
def setup_logger(name='Gateway'):
    log_dir = Path('logs/gateway')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f'{name.lower()}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
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

logger, _ = setup_logger()


# ============ 共享数据 ============
@dataclass
class FrameData:
    """帧数据"""
    frame_id: str
    timestamp: float
    image_data: bytes
    client_id: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class ResultData:
    """结果数据"""
    frame_id: str
    timestamp: float
    detections: List[Dict]
    processing_time_ms: float
    client_id: str


# ============ gRPC 客户端封装 ============
class GRPCClient:
    """gRPC客户端 - 连接视觉处理服务"""
    
    def __init__(self, server_address: str = 'localhost:50051'):
        self.server_address = server_address
        self.channel = None
        self.stub = None
        self._connect()
    
    def _connect(self):
        """连接gRPC服务"""
        try:
            import grpc_stream_pb2
            import grpc_stream_pb2_grpc
            
            self.channel = grpc.insecure_channel(self.server_address)
            self.stub = grpc_stream_pb2_grpc.VisionStreamStub(self.channel)
            logger.info(f"✓ gRPC客户端连接到: {self.server_address}")
        except ImportError:
            logger.error("请先生成gRPC代码: python -m grpc_tools.protoc ...")
        except Exception as e:
            logger.error(f"gRPC连接失败: {e}")
    
    def process_frame(self, frame: FrameData) -> Optional[ResultData]:
        """处理单帧"""
        if not self.stub:
            return None
        
        try:
            import grpc_stream_pb2
            
            # 构建请求
            request = grpc_stream_pb2.ImageFrame(
                frame_id=frame.frame_id,
                timestamp=int(frame.timestamp * 1000),
                image=grpc_stream_pb2.ImageData(
                    data=frame.image_data,
                    format='jpeg'
                ),
                metadata=frame.metadata
            )
            
            # 调用gRPC
            response = self.stub.ProcessImage(request, timeout=10.0)
            
            # 转换结果
            detections = []
            for det in response.detections:
                detections.append({
                    'type': det.type,
                    'class_name': det.class_name,
                    'confidence': det.confidence,
                    'bbox': [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2],
                    'area': det.area,
                })
            
            return ResultData(
                frame_id=response.frame_id,
                timestamp=response.timestamp / 1000,
                detections=detections,
                processing_time_ms=response.processing_time_ms,
                client_id=frame.client_id
            )
        
        except grpc.RpcError as e:
            logger.error(f"gRPC调用失败: {e}")
            return None
    
    def process_stream(self, frames: List[FrameData]) -> List[ResultData]:
        """批量处理"""
        results = []
        for frame in frames:
            result = self.process_frame(frame)
            if result:
                results.append(result)
        return results
    
    def close(self):
        """关闭连接"""
        if self.channel:
            self.channel.close()


# ============ gRPC 服务端封装 ============
class GRPCProcessor:
    """gRPC服务端 - 视觉处理服务"""
    
    def __init__(self, model_path: str = 'models/yolov8/train/weights/best.pt'):
        self.model = None
        self.stats = {'processed': 0, 'errors': 0, 'total_time_ms': 0}
        self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """加载模型"""
        if not Path(model_path).exists():
            logger.warning(f"模型不存在: {model_path}")
            return
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            logger.info(f"✓ YOLO模型加载: {model_path}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
    
    def detect(self, image_data: bytes, conf: float = 0.25, iou: float = 0.45) -> Dict:
        """检测图像"""
        start_time = time.time()
        
        try:
            # 解码
            np_arr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("图像解码失败")
            
            detections = []
            
            if self.model:
                results = self.model(image, conf=conf, iou=iou, verbose=False)
                
                if results and results[0].boxes is not None:
                    class_names = {
                        0: 'missing_hole', 1: 'mouse_bite', 2: 'open_circuit',
                        3: 'short', 4: 'spur', 5: 'spurious_copper', 6: 'normal'
                    }
                    
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections.append({
                            'type': 'defect',
                            'class_id': int(box.cls[0]),
                            'class_name': class_names.get(int(box.cls[0]), 'unknown'),
                            'confidence': float(box.conf[0]),
                            'bbox': {
                                'x1': int(x1), 'y1': int(y1),
                                'x2': int(x2), 'y2': int(y2)
                            },
                            'center': {
                                'x': int((x1+x2)/2),
                                'y': int((y1+y2)/2)
                            }
                        })
            
            processing_time = (time.time() - start_time) * 1000
            self.stats['processed'] += 1
            self.stats['total_time_ms'] += processing_time
            
            return {
                'detections': detections,
                'processing_time_ms': processing_time,
                'detection_count': len(detections)
            }
        
        except Exception as e:
            logger.error(f"检测失败: {e}")
            self.stats['errors'] += 1
            return {'detections': [], 'processing_time_ms': 0, 'error': str(e)}


# ============ WebSocket 网关 ============
class WebSocketGateway:
    """WebSocket网关 - 前端通信"""
    
    def __init__(self, port: int = 8765, grpc_client: GRPCClient = None):
        self.port = port
        self.grpc_client = grpc_client
        self.clients: Dict[str, 'ClientConnection'] = {}
        self.running = False
        self._server = None
    
    async def handle_client(self, websocket, path):
        """处理WebSocket客户端"""
        client_id = f"ws_{len(self.clients)}_{int(time.time())}"
        
        # 创建客户端连接对象
        conn = ClientConnection(client_id, websocket, self.grpc_client)
        self.clients[client_id] = conn
        
        logger.info(f"[{client_id}] 客户端连接 ({path})")
        
        try:
            await websocket.send(json.dumps({
                'type': 'connected',
                'client_id': client_id,
                'mode': 'grpc+websocket'
            }))
            
            async for message in websocket:
                await conn.handle_message(message)
        
        except Exception as e:
            logger.info(f"[{client_id}] 连接关闭: {e}")
        finally:
            del self.clients[client_id]
    
    async def start(self):
        """启动WebSocket服务器"""
        try:
            import websockets
            
            logger.info("=" * 50)
            logger.info(f"WebSocket 网关启动")
            logger.info(f"地址: ws://0.0.0.0:{self.port}")
            logger.info(f"gRPC后端: {self.grpc_client.server_address if self.grpc_client else '未连接'}")
            logger.info("=" * 50)
            
            async with websockets.serve(self.handle_client, '0.0.0.0', self.port):
                await asyncio.Future()
        
        except ImportError:
            logger.error("需要安装websockets: pip install websockets")
        except Exception as e:
            logger.error(f"WebSocket启动失败: {e}")


class ClientConnection:
    """客户端连接"""
    
    def __init__(self, client_id: str, websocket, grpc_client: GRPCClient):
        self.client_id = client_id
        self.ws = websocket
        self.grpc_client = grpc_client
        self.frame_count = 0
        self.last_stats_time = time.time()
        self.fps = 0
    
    async def handle_message(self, message):
        """处理消息"""
        try:
            if isinstance(message, bytes):
                # 二进制图像数据
                frame_id = f"{self.client_id}_{self.frame_count}"
                self.frame_count += 1
                
                frame = FrameData(
                    frame_id=frame_id,
                    timestamp=time.time(),
                    image_data=message,
                    client_id=self.client_id
                )
                
                # 通过gRPC处理
                result = self.grpc_client.process_frame(frame)
                
                if result:
                    await self.send_result(result)
                    self._update_fps()
            
            else:
                # JSON消息
                data = json.loads(message)
                msg_type = data.get('type')
                
                if msg_type == 'image':
                    # Base64图像
                    img_data = base64.b64decode(data.get('data', ''))
                    frame_id = data.get('frame_id', f"{self.client_id}_{self.frame_count}")
                    self.frame_count += 1
                    
                    frame = FrameData(
                        frame_id=frame_id,
                        timestamp=time.time(),
                        image_data=img_data,
                        client_id=self.client_id,
                        metadata=data.get('metadata', {})
                    )
                    
                    result = self.grpc_client.process_frame(frame)
                    
                    if result:
                        await self.send_result(result)
                        self._update_fps()
                
                elif msg_type == 'batch':
                    # 批量处理
                    frames = []
                    for i, img_base64 in enumerate(data.get('images', [])):
                        img_data = base64.b64decode(img_base64)
                        frame = FrameData(
                            frame_id=f"{self.client_id}_batch_{i}",
                            timestamp=time.time(),
                            image_data=img_data,
                            client_id=self.client_id
                        )
                        frames.append(frame)
                    
                    results = self.grpc_client.process_stream(frames)
                    
                    await self.ws.send(json.dumps({
                        'type': 'batch_result',
                        'results': [r.__dict__ for r in results]
                    }))
                
                elif msg_type == 'ping':
                    await self.ws.send(json.dumps({'type': 'pong'}))
                
                elif msg_type == 'stats':
                    await self.ws.send(json.dumps({
                        'type': 'stats',
                        'fps': self.fps,
                        'frames_sent': self.frame_count
                    }))
        
        except json.JSONDecodeError:
            logger.error(f"[{self.client_id}] JSON解析失败")
        except Exception as e:
            logger.error(f"[{self.client_id}] 处理错误: {e}")
    
    async def send_result(self, result: ResultData):
        """发送结果"""
        response = {
            'type': 'result',
            'frame_id': result.frame_id,
            'timestamp': result.timestamp,
            'detections': result.detections,
            'detection_count': len(result.detections),
            'processing_time_ms': result.processing_time_ms,
            'fps': self.fps
        }
        await self.ws.send(json.dumps(response))
    
    def _update_fps(self):
        """更新FPS"""
        now = time.time()
        if now - self.last_stats_time >= 1.0:
            self.fps = self.frame_count / (now - self.last_stats_time)
            self.frame_count = 0
            self.last_stats_time = now


# ============ HTTP 网关 (备选) ============
class HTTPGateway:
    """HTTP网关 - REST API"""
    
    def __init__(self, grpc_client: GRPCClient = None, port: int = 8080):
        self.grpc_client = grpc_client
        self.port = port
        self.app = None
    
    async def handle_request(self, request):
        """处理HTTP请求"""
        from aiohttp import web
        
        if request.path == '/health':
            return web.json_response({'status': 'ok', 'mode': 'grpc+http'})
        
        elif request.path == '/detect' and request.method == 'POST':
            # 接收图像
            data = await request.read()
            
            frame = FrameData(
                frame_id=f"http_{int(time.time())}",
                timestamp=time.time(),
                image_data=data,
                client_id='http_client'
            )
            
            result = self.grpc_client.process_frame(frame)
            
            if result:
                return web.json_response(result.__dict__)
            else:
                return web.json_response({'error': 'processing failed'}, status=500)
        
        elif request.path == '/detect_batch' and request.method == 'POST':
            # 批量检测
            body = await request.json()
            frames = []
            
            for i, img_base64 in enumerate(body.get('images', [])):
                img_data = base64.b64decode(img_base64)
                frame = FrameData(
                    frame_id=f"http_batch_{i}",
                    timestamp=time.time(),
                    image_data=img_data,
                    client_id='http_client'
                )
                frames.append(frame)
            
            results = self.grpc_client.process_stream(frames)
            
            return web.json_response({
                'results': [r.__dict__ for r in results],
                'count': len(results)
            })
        
        return web.json_response({'error': 'not found'}, status=404)
    
    async def start(self):
        """启动HTTP服务器"""
        try:
            from aiohttp import web
            
            self.app = web.Application()
            self.app.router.add_route('*', '/{tail:.*}', self.handle_request)
            
            runner = web.AppRunner(self.app)
            await runner.setup()
            site = web.TCPSite(runner, '0.0.0.0', self.port)
            await site.start()
            
            logger.info("=" * 50)
            logger.info(f"HTTP 网关启动")
            logger.info(f"地址: http://0.0.0.0:{self.port}")
            logger.info(f"端点:")
            logger.info(f"  POST /detect - 单图检测")
            logger.info(f"  POST /detect_batch - 批量检测")
            logger.info(f"  GET /health - 健康检查")
            logger.info("=" * 50)
            
            await asyncio.Future()
        
        except ImportError:
            logger.error("需要安装aiohttp: pip install aiohttp")


# ============ 统一启动器 ============
async def run_gateway(ws_port: int = 8765, http_port: int = 8080, grpc_address: str = 'localhost:50051'):
    """运行网关服务"""
    logger.info("启动网关服务...")
    
    # 创建gRPC客户端
    grpc_client = GRPCClient(grpc_address)
    
    # WebSocket网关
    ws_gateway = WebSocketGateway(ws_port, grpc_client)
    
    # HTTP网关
    http_gateway = HTTPGateway(grpc_client, http_port)
    
    # 同时启动两个网关
    ws_task = asyncio.create_task(ws_gateway.start())
    http_task = asyncio.create_task(http_gateway.start())
    
    logger.info("网关服务已启动，按 Ctrl+C 停止")
    
    await asyncio.gather(ws_task, http_task)


def run_processor(grpc_port: int = 50051, model_path: str = 'models/yolov8/train/weights/best.pt'):
    """运行视觉处理服务"""
    logger.info("启动视觉处理服务...")
    
    try:
        import grpc_stream_pb2
        import grpc_stream_pb2_grpc
        from concurrent import futures
    except ImportError:
        logger.error("请先生成gRPC代码:")
        logger.error("  python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. grpc_stream.proto")
        return
    
    # 创建处理器
    processor = GRPCProcessor(model_path)
    
    # 创建gRPC服务
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # 简单服务实现
    class SimpleServicer(grpc_stream_pb2_grpc.VisionStreamServicer):
        def ProcessImage(self, request, context):
            result = processor.detect(request.image.data)
            return grpc_stream_pb2.ImageFrame(
                frame_id=request.frame_id,
                timestamp=request.timestamp,
                detections=[],
                processing_time_ms=result['processing_time_ms']
            )
        
        def ProcessStream(self, request_iterator, context):
            for request in request_iterator:
                result = processor.detect(request.image.data)
                yield grpc_stream_pb2.ImageFrame(
                    frame_id=request.frame_id,
                    timestamp=request.timestamp,
                    detections=[],
                    processing_time_ms=result['processing_time_ms']
                )
    
    grpc_stream_pb2_grpc.add_VisionStreamServicer_to_server(
        SimpleServicer(), server
    )
    
    server.add_insecure_port(f'[::]:{grpc_port}')
    server.start()
    
    logger.info("=" * 50)
    logger.info(f"视觉处理服务启动")
    logger.info(f"gRPC地址: [::]:{grpc_port}")
    logger.info(f"模型: {model_path}")
    logger.info(f"已处理: {processor.stats['processed']}, 错误: {processor.stats['errors']}")
    logger.info("=" * 50)
    
    try:
        while True:
            time.sleep(1)
            if processor.stats['processed'] > 0:
                avg_time = processor.stats['total_time_ms'] / processor.stats['processed']
                print(f"\r已处理: {processor.stats['processed']} | "
                      f"错误: {processor.stats['errors']} | "
                      f"平均耗时: {avg_time:.1f}ms", end='')
    except KeyboardInterrupt:
        server.stop(grace=5)
        logger.info("\n服务已停止")


# ============ 主函数 ============
def main():
    parser = argparse.ArgumentParser(description='gRPC + WebSocket 网关服务')
    
    parser.add_argument('--mode', type=str, default='gateway',
                       choices=['gateway', 'processor', 'all'],
                       help='运行模式: gateway=网关, processor=处理服务, all=全部')
    
    # 端口配置
    parser.add_argument('--ws-port', type=int, default=8765, help='WebSocket端口')
    parser.add_argument('--http-port', type=int, default=8080, help='HTTP端口')
    parser.add_argument('--grpc-port', type=int, default=50051, help='gRPC端口')
    
    # gRPC配置
    parser.add_argument('--grpc-address', type=str, default='localhost:50051',
                       help='gRPC服务器地址')
    
    # 模型配置
    parser.add_argument('--model', type=str, 
                       default='models/yolov8/train/weights/best.pt',
                       help='模型路径')
    
    args = parser.parse_args()
    
    if args.mode == 'gateway':
        # 只运行网关
        asyncio.run(run_gateway(
            ws_port=args.ws_port,
            http_port=args.http_port,
            grpc_address=args.grpc_address
        ))
    
    elif args.mode == 'processor':
        # 只运行处理服务
        run_processor(grpc_port=args.grpc_port, model_path=args.model)
    
    elif args.mode == 'all':
        # 同时运行
        import multiprocessing
        
        # 在新进程中运行处理服务
        p = multiprocessing.Process(
            target=run_processor,
            args=(args.grpc_port, args.model)
        )
        p.start()
        
        # 在主进程中运行网关
        time.sleep(2)  # 等待处理服务启动
        asyncio.run(run_gateway(
            ws_port=args.ws_port,
            http_port=args.http_port,
            grpc_address=f'localhost:{args.grpc_port}'
        ))
        
        p.terminate()


if __name__ == '__main__':
    main()
