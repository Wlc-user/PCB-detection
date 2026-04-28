"""
gRPC + WebSocket 客户端
测试图像发送和结果接收

使用方式:
    python grpc_websocket_client.py --mode ws --image test.jpg
    python grpc_websocket_client.py --mode grpc --image test.jpg
"""

import os
import sys
import time
import asyncio
import argparse
import base64
import logging
from pathlib import Path

import cv2
import numpy as np
import grpc
import websockets
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============ WebSocket 客户端 ============
class WSClient:
    """WebSocket客户端"""
    
    def __init__(self, uri: str = 'ws://localhost:8765'):
        self.uri = uri
        self.ws = None
        self.frame_count = 0
    
    async def connect(self):
        """连接服务器"""
        self.ws = await websockets.connect(self.uri)
        logger.info(f"✓ 已连接: {self.uri}")
        
        # 接收欢迎消息
        welcome = await self.ws.recv()
        logger.info(f"服务器: {welcome}")
    
    async def send_image(self, image_path: str) -> dict:
        """发送图像"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 编码为JPEG
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        img_base64 = base64.b64encode(buffer).decode()
        
        # 构建消息
        self.frame_count += 1
        message = {
            'type': 'image',
            'frame_id': f'client_{self.frame_count}',
            'timestamp': time.time(),
            'data': img_base64,
            'metadata': {'source': 'test_client'}
        }
        
        # 发送
        start = time.time()
        await self.ws.send(message)
        
        # 等待结果
        response = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
        elapsed = (time.time() - start) * 1000
        
        return {'response': response, 'time_ms': elapsed}
    
    async def send_batch(self, image_paths: list) -> list:
        """批量发送"""
        images = []
        for path in image_paths:
            image = cv2.imread(path)
            if image is not None:
                _, buffer = cv2.imencode('.jpg', image)
                images.append(base64.b64encode(buffer).decode())
        
        message = {
            'type': 'batch',
            'images': images
        }
        
        start = time.time()
        await self.ws.send(message)
        
        response = await asyncio.wait_for(self.ws.recv(), timeout=60.0)
        elapsed = (time.time() - start) * 1000
        
        return {'response': response, 'time_ms': elapsed}
    
    async def close(self):
        """关闭"""
        if self.ws:
            await self.ws.close()
            logger.info("连接已关闭")


# ============ gRPC 客户端 ============
class GRPCClient:
    """gRPC客户端"""
    
    def __init__(self, address: str = 'localhost:50051'):
        self.address = address
        self.channel = None
        self.stub = None
        self._connect()
    
    def _connect(self):
        """连接"""
        try:
            import grpc_stream_pb2
            import grpc_stream_pb2_grpc
            
            self.channel = grpc.insecure_channel(self.address)
            self.stub = grpc_stream_pb2_grpc.VisionStreamStub(self.channel)
            logger.info(f"✓ gRPC连接: {self.address}")
        except ImportError:
            logger.error("请先生成gRPC代码")
        except Exception as e:
            logger.error(f"连接失败: {e}")
    
    def detect(self, image_path: str) -> dict:
        """检测图像"""
        if not self.stub:
            return {'error': 'not connected'}
        
        import grpc_stream_pb2
        
        # 读取图像
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # 构建请求
        request = grpc_stream_pb2.ImageData(
            data=image_data,
            format='jpeg'
        )
        
        # 调用
        start = time.time()
        try:
            response = self.stub.ProcessImage(request, timeout=30.0)
            elapsed = (time.time() - start) * 1000
            
            detections = []
            for det in response.detections:
                detections.append({
                    'class_name': det.class_name,
                    'confidence': det.confidence,
                    'bbox': [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2]
                })
            
            return {
                'frame_id': response.frame_id,
                'detections': detections,
                'processing_time_ms': elapsed,
                'detection_count': len(detections)
            }
        except grpc.RpcError as e:
            return {'error': str(e)}


# ============ HTTP 客户端 ============
class HTTPClient:
    """HTTP客户端"""
    
    def __init__(self, base_url: str = 'http://localhost:8080'):
        self.base_url = base_url
    
    async def detect(self, image_path: str) -> dict:
        """检测图像"""
        # 读取图像
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # 编码为Base64
        img_base64 = base64.b64encode(image_data).decode()
        
        # 发送请求
        async with aiohttp.ClientSession() as session:
            start = time.time()
            async with session.post(
                f'{self.base_url}/detect',
                json={'data': img_base64},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                result = await resp.json()
                result['time_ms'] = (time.time() - start) * 1000
                return result


# ============ 主函数 ============
async def test_websocket(image_path: str):
    """测试WebSocket"""
    client = WSClient()
    await client.connect()
    
    logger.info(f"发送图像: {image_path}")
    result = await client.send_image(image_path)
    
    logger.info("=" * 40)
    logger.info("检测结果:")
    logger.info(f"  耗时: {result['time_ms']:.1f}ms")
    
    response = result['response']
    if isinstance(response, str):
        data = eval(response) if response.startswith('{') else response
        if isinstance(data, dict):
            for det in data.get('detections', [])[:5]:
                logger.info(f"  - {det.get('class_name', 'N/A')}: "
                          f"{det.get('confidence', 0):.3f}")
    
    await client.close()


def test_grpc(image_path: str):
    """测试gRPC"""
    client = GRPCClient()
    
    logger.info(f"发送图像: {image_path}")
    result = client.detect(image_path)
    
    logger.info("=" * 40)
    logger.info("检测结果:")
    
    if 'error' in result:
        logger.error(f"  错误: {result['error']}")
    else:
        logger.info(f"  耗时: {result.get('time_ms', result.get('processing_time_ms', 0)):.1f}ms")
        logger.info(f"  检测数: {result.get('detection_count', len(result.get('detections', [])))}")
        
        for det in result.get('detections', [])[:5]:
            logger.info(f"  - {det.get('class_name', 'N/A')}: {det.get('confidence', 0):.3f}")


async def test_http(image_path: str):
    """测试HTTP"""
    client = HTTPClient()
    
    logger.info(f"发送图像: {image_path}")
    result = await client.detect(image_path)
    
    logger.info("=" * 40)
    logger.info("检测结果:")
    logger.info(f"  耗时: {result.get('time_ms', 0):.1f}ms")
    
    for det in result.get('detections', [])[:5]:
        logger.info(f"  - {det.get('class_name', 'N/A')}: {det.get('confidence', 0):.3f}")


def main():
    parser = argparse.ArgumentParser(description='gRPC + WebSocket 测试客户端')
    
    parser.add_argument('--mode', type=str, default='ws',
                       choices=['ws', 'grpc', 'http'],
                       help='测试模式')
    parser.add_argument('--image', type=str, required=True,
                       help='测试图像路径')
    parser.add_argument('--uri', type=str, default='ws://localhost:8765',
                       help='WebSocket地址')
    parser.add_argument('--grpc-addr', type=str, default='localhost:50051',
                       help='gRPC地址')
    parser.add_argument('--http-url', type=str, default='http://localhost:8080',
                       help='HTTP地址')
    
    args = parser.parse_args()
    
    if args.mode == 'ws':
        asyncio.run(test_websocket(args.image))
    elif args.mode == 'grpc':
        test_grpc(args.image)
    elif args.mode == 'http':
        asyncio.run(test_http(args.image))


if __name__ == '__main__':
    main()
