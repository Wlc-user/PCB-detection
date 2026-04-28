"""
视频流 + 大模型 + MCP 协议集成服务

功能:
    - 实时视频流采集 (RTSP/摄像头/文件)
    - YOLO 缺陷检测
    - 大模型视频理解 (Qwen/VL/其他)
    - MCP 协议传输 (文档格式音频聊天)

使用方式:
    python video_llm_mcp_server.py --mode camera
    python video_llm_mcp_server.py --mode rtsp --url rtsp://xxx
    python video_llm_mcp_server.py --mode file --path video.mp4
"""

import os
import sys
import time
import asyncio
import base64
import json
import queue
import threading
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import io
import tempfile

import cv2
import numpy as np

# ============ 日志配置 ============
def setup_logger(name='VideoLLM', log_dir='logs/video_llm', level=logging.INFO):
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_path / f'video_llm_{timestamp}.log'
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, log_file

logger, log_file = setup_logger()


# ============ MCP 协议定义 ============
@dataclass
class MCPMessage:
    """MCP 协议消息"""
    msg_id: str
    msg_type: str  # text, audio_transcript, detection_report, chat_response
    timestamp: float
    content: str
    metadata: Dict = field(default_factory=dict)
    
    def to_json(self) -> str:
        return json.dumps({
            'msg_id': self.msg_id,
            'msg_type': self.msg_type,
            'timestamp': self.timestamp,
            'content': self.content,
            'metadata': self.metadata
        }, ensure_ascii=False)


class MCPProtocol:
    """MCP 协议处理器"""
    
    # 消息类型
    TYPE_DETECTION_REPORT = "detection_report"      # 检测报告
    TYPE_CHAT_RESPONSE = "chat_response"              # 聊天响应
    TYPE_AUDIO_TRANSCRIPT = "audio_transcript"        # 音频转写
    TYPE_STREAM_FRAME = "stream_frame"                 # 流帧
    TYPE_SYSTEM_STATUS = "system_status"              # 系统状态
    
    def __init__(self):
        self.msg_counter = 0
    
    def new_msg_id(self) -> str:
        self.msg_counter += 1
        return f"mcp_{datetime.now().strftime('%H%M%S')}_{self.msg_counter}"
    
    def create_detection_report(self, detections: List[Dict], 
                                frame_info: Dict = None) -> MCPMessage:
        """创建检测报告 (文档格式)"""
        summary = self._generate_detection_summary(detections)
        
        return MCPMessage(
            msg_id=self.new_msg_id(),
            msg_type=self.TYPE_DETECTION_REPORT,
            timestamp=time.time(),
            content=summary,
            metadata={
                'detections': detections,
                'count': len(detections),
                'frame_info': frame_info or {}
            }
        )
    
    def create_chat_response(self, text: str, 
                             audio_transcript: str = None) -> MCPMessage:
        """创建聊天响应 (文档格式)"""
        return MCPMessage(
            msg_id=self.new_msg_id(),
            msg_type=self.TYPE_CHAT_RESPONSE,
            timestamp=time.time(),
            content=text,
            metadata={
                'audio_transcript': audio_transcript,
                'format': 'document'
            }
        )
    
    def create_audio_transcript(self, text: str) -> MCPMessage:
        """创建音频转写"""
        return MCPMessage(
            msg_id=self.new_msg_id(),
            msg_type=self.TYPE_AUDIO_TRANSCRIPT,
            timestamp=time.time(),
            content=text,
            metadata={'source': 'asr'}
        )
    
    def _generate_detection_summary(self, detections: List[Dict]) -> str:
        """生成检测报告 (文档格式)"""
        if not detections:
            return "## 检测报告\n\n未检测到目标\n"
        
        # 按类型分组
        by_class = {}
        for det in detections:
            cls_name = det.get('class_name', 'unknown')
            if cls_name not in by_class:
                by_class[cls_name] = []
            by_class[cls_name].append(det)
        
        lines = [
            "## 缺陷检测报告",
            "",
            f"**检测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**目标总数**: {len(detections)}",
            "",
            "---",
            "",
            "### 检测详情",
            ""
        ]
        
        for cls_name, items in by_class.items():
            lines.append(f"#### {cls_name} ({len(items)}个)")
            for i, item in enumerate(items, 1):
                conf = item.get('confidence', 0)
                bbox = item.get('bbox', [])
                center = item.get('center', [])
                lines.append(f"- **#{i}** 置信度: {conf:.2%}")
                if bbox:
                    lines.append(f"  - 位置: ({bbox[0]}, {bbox[1]}) - ({bbox[2]}, {bbox[3]})")
                if center:
                    lines.append(f"  - 中心点: ({center[0]}, {center[1]})")
            lines.append("")
        
        return "\n".join(lines)


# ============ YOLO 检测器 ============
class YOLODetector:
    """YOLO PCB 缺陷检测器"""
    
    CLASS_NAMES = {
        0: 'missing_hole',      # 缺孔
        1: 'mouse_bite',        # 鼠咬
        2: 'open_circuit',      # 开路
        3: 'short',             # 短路
        4: 'spur',              # 毛刺
        5: 'spurious_copper',   # 多铜
        6: 'normal'             # 正常
    }
    
    def __init__(self, model_path: str = 'models/yolov8/train/weights/best.pt',
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        if not Path(self.model_path).exists():
            logger.warning(f"YOLO模型不存在: {self.model_path}")
            return
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            logger.info(f"YOLO模型加载: {self.model_path}")
        except Exception as e:
            logger.error(f"YOLO加载失败: {e}")
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """检测缺陷"""
        if self.model is None:
            return []
        
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            imgsz=640
        )
        
        detections = []
        if results and len(results) > 0:
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        
                        detections.append({
                            'class_id': cls_id,
                            'class_name': self.CLASS_NAMES.get(cls_id, 'unknown'),
                            'confidence': round(conf, 4),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'area': int((x2-x1) * (y2-y1)),
                            'center': [int((x1+x2)/2), int((y1+y2)/2)]
                        })
        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """绘制检测结果"""
        result = image.copy()
        
        color_map = {
            'missing_hole': (255, 0, 0),      # 蓝色
            'mouse_bite': (0, 255, 0),         # 绿色
            'open_circuit': (0, 0, 255),       # 红色
            'short': (255, 255, 0),            # 青色
            'spur': (255, 0, 255),             # 紫色
            'spurious_copper': (0, 255, 255),  # 黄色
            'normal': (128, 128, 128)          # 灰色
        }
        
        for det in detections:
            bbox = det['bbox']
            color = color_map.get(det['class_name'], (0, 255, 0))
            conf = det['confidence']
            label = f"{det['class_name']}: {conf:.2f}"
            
            # 画框
            cv2.rectangle(result, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # 画标签背景
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (bbox[0], bbox[1]-h-10), (bbox[0]+w, bbox[1]), color, -1)
            
            # 写文字
            cv2.putText(result, label, (bbox[0], bbox[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result


# ============ 大模型接口 ============
class LLMInterface:
    """大模型接口 (支持多种后端)"""
    
    def __init__(self, provider: str = 'openai', api_key: str = None,
                 model: str = 'gpt-4o', base_url: str = None):
        self.provider = provider
        self.api_key = api_key or os.getenv('OPENAI_API_KEY', '')
        self.model = model
        self.base_url = base_url
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """初始化客户端"""
        if not self.api_key:
            logger.warning("未设置 API Key，使用模拟模式")
            return
        
        try:
            if self.provider == 'openai':
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            elif self.provider == 'qwen':
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key, 
                                    base_url=self.base_url or 'https://dashscope.aliyuncs.com/compatible-mode/v1')
            elif self.provider == 'zhipu':
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key,
                                    base_url=self.base_url or 'https://open.bigmodel.cn/api/paas/v4')
            logger.info(f"LLM客户端初始化: {self.provider}/{self.model}")
        except Exception as e:
            logger.error(f"LLM客户端初始化失败: {e}")
    
    async def analyze_video_frame(self, image: np.ndarray, 
                                   prompt: str = None) -> str:
        """分析视频帧"""
        if self.client is None:
            return self._mock_analysis(image)
        
        default_prompt = (
            "请分析这张PCB电路板图像，描述你看到的任何缺陷或异常。 "
            "如果发现问题，请详细说明缺陷类型、位置和严重程度。"
        )
        
        try:
            # 图像转 base64
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_base64 = base64.b64encode(buffer).decode()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt or default_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                        ]
                    }
                ],
                max_tokens=500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"视频分析失败: {e}")
            return self._mock_analysis(image)
    
    def chat(self, text: str, system_prompt: str = None) -> str:
        """文本对话"""
        if self.client is None:
            return f"[模拟回复] 收到消息: {text[:50]}..."
        
        default_system = "你是一个专业的PCB缺陷检测助手，请用简洁专业的语言回答。"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt or default_system},
                    {"role": "user", "content": text}
                ],
                max_tokens=300
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"对话失败: {e}")
            return f"处理失败: {str(e)}"
    
    def _mock_analysis(self, image: np.ndarray) -> str:
        """模拟分析结果"""
        return f"[模拟分析] 图像尺寸: {image.shape[:2]}"


# ============ 音频转写 (ASR) ============
class ASRInterface:
    """语音识别接口"""
    
    def __init__(self, provider: str = 'mock'):
        self.provider = provider
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """初始化客户端"""
        if self.provider == 'azure':
            # Azure Speech Services
            try:
                import azure.cognitiveservices.speech as speech_sdk
                # 配置 Azure Speech
                logger.info("ASR客户端: Azure")
            except ImportError:
                logger.warning("Azure SDK 未安装")
        elif self.provider == 'vosk':
            try:
                from vosk import Model, KaldiRecognizer
                # 加载 Vosk 模型
                logger.info("ASR客户端: Vosk")
            except ImportError:
                logger.warning("Vosk 未安装")
        else:
            logger.info("ASR客户端: 模拟模式")
    
    def transcribe(self, audio_data: bytes) -> str:
        """转写音频"""
        # 模拟转写
        return "[模拟转写] 这是一条测试语音转文字的结果"
    
    def transcribe_from_file(self, audio_path: str) -> str:
        """从文件转写"""
        try:
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            return self.transcribe(audio_data)
        except Exception as e:
            logger.error(f"音频转写失败: {e}")
            return ""


# ============ 视频流采集器 ============
class VideoStreamCapture:
    """视频流采集器"""
    
    def __init__(self, source, fps: int = 30):
        self.source = source
        self.fps = fps
        self.cap = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.capture_thread = None
    
    def start(self):
        """启动采集"""
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source)
        elif self.source.startswith(('rtsp://', 'rtmp://', 'http://')):
            self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            logger.error(f"无法打开视频源: {self.source}")
            return False
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info(f"视频流采集启动: {self.source}")
        return True
    
    def _capture_loop(self):
        """采集循环"""
        frame_count = 0
        delay = 1.0 / self.fps if self.fps > 0 else 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("视频流断开，尝试重连...")
                time.sleep(1)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.source)
                continue
            
            frame_count += 1
            
            try:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put_nowait((frame_count, frame.copy()))
            except queue.Full:
                pass
            
            if delay > 0:
                time.sleep(delay)
    
    def read(self) -> Tuple[int, np.ndarray]:
        """读取帧"""
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return 0, None
    
    def stop(self):
        """停止采集"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        logger.info("视频流采集停止")


# ============ 主服务 ============
class VideoLLMMCPService:
    """视频流 + LLM + MCP 服务"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # 初始化组件
        self.detector = YOLODetector(
            model_path=self.config.get('yolo_model', 'models/yolov8/train/weights/best.pt'),
            conf_threshold=self.config.get('conf_threshold', 0.25),
            iou_threshold=self.config.get('iou_threshold', 0.45)
        )
        
        self.llm = LLMInterface(
            provider=self.config.get('llm_provider', 'openai'),
            api_key=self.config.get('api_key', ''),
            model=self.config.get('llm_model', 'gpt-4o'),
            base_url=self.config.get('llm_base_url', None)
        )
        
        self.asr = ASRInterface(
            provider=self.config.get('asr_provider', 'mock')
        )
        
        self.mcp = MCPProtocol()
        
        # 流处理
        self.stream_capture = None
        self.processing = False
        self.stats = {
            'frames_processed': 0,
            'detections_count': 0,
            'llm_calls': 0,
            'mcp_messages': 0,
            'fps': 0,
            'start_time': time.time()
        }
    
    async def process_frame(self, frame: np.ndarray) -> MCPMessage:
        """处理单帧"""
        # 1. YOLO 检测
        detections = self.detector.detect(frame)
        
        # 2. 生成检测报告
        report = self.mcp.create_detection_report(
            detections=detections,
            frame_info={'shape': frame.shape}
        )
        
        # 3. 如果有检测结果，调用 LLM 分析
        if detections:
            llm_analysis = await self.llm.analyze_video_frame(frame)
            if llm_analysis:
                chat_response = self.mcp.create_chat_response(
                    text=llm_analysis,
                    audio_transcript=None
                )
                self.stats['llm_calls'] += 1
                self.stats['mcp_messages'] += 1
                return chat_response
        
        self.stats['detections_count'] += len(detections)
        self.stats['mcp_messages'] += 1
        return report
    
    def process_audio(self, audio_data: bytes) -> MCPMessage:
        """处理音频"""
        # 1. ASR 转写
        transcript = self.asr.transcribe(audio_data)
        
        # 2. 创建音频转写消息
        transcript_msg = self.mcp.create_audio_transcript(transcript)
        
        # 3. LLM 生成回复
        llm_response = self.llm.chat(transcript)
        
        # 4. 创建聊天响应消息
        chat_msg = self.mcp.create_chat_response(
            text=llm_response,
            audio_transcript=transcript
        )
        
        self.stats['mcp_messages'] += 2
        return chat_msg
    
    async def run_camera_mode(self, camera_id: int = 0, fps: int = 10):
        """摄像头模式"""
        logger.info(f"启动摄像头模式 (ID: {camera_id}, FPS: {fps})")
        
        self.stream_capture = VideoStreamCapture(camera_id, fps=fps)
        if not self.stream_capture.start():
            return
        
        last_process_time = time.time()
        process_interval = 1.0 / fps if fps > 0 else 0.1
        
        frame_id = 0
        last_stats_time = time.time()
        
        try:
            while self.processing:
                frame_num, frame = self.stream_capture.read()
                if frame is None:
                    continue
                
                # 定期处理帧
                current_time = time.time()
                if current_time - last_process_time >= process_interval:
                    frame_id += 1
                    last_process_time = current_time
                    
                    # 处理帧
                    result = await self.process_frame(frame)
                    self.stats['frames_processed'] += 1
                    
                    # 打印 MCP 消息
                    if frame_id % 10 == 0:
                        logger.info(f"[帧 {frame_id}] MCP消息: {result.msg_type}")
                        logger.info(f"  内容预览: {result.content[:100]}...")
                
                # 定期更新 FPS
                if current_time - last_stats_time >= 5.0:
                    elapsed = current_time - self.stats['start_time']
                    self.stats['fps'] = self.stats['frames_processed'] / elapsed
                    logger.info(f"状态: FPS={self.stats['fps']:.1f}, "
                               f"检测={self.stats['detections_count']}, "
                               f"LLM调用={self.stats['llm_calls']}")
                    last_stats_time = current_time
                
                # 显示预览
                cv2.imshow('Video LLM MCP', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            logger.info("收到中断信号")
        finally:
            self.stream_capture.stop()
            cv2.destroyAllWindows()
    
    async def run_rtsp_mode(self, rtsp_url: str, fps: int = 10):
        """RTSP 流模式"""
        logger.info(f"启动 RTSP 模式: {rtsp_url}")
        
        self.stream_capture = VideoStreamCapture(rtsp_url, fps=fps)
        if not self.stream_capture.start():
            return
        
        await self._run_stream_processing(fps)
    
    async def run_file_mode(self, video_path: str, fps: int = 30):
        """视频文件模式"""
        logger.info(f"启动文件模式: {video_path}")
        
        self.stream_capture = VideoStreamCapture(video_path, fps=fps)
        if not self.stream_capture.start():
            return
        
        await self._run_stream_processing(fps)
    
    async def _run_stream_processing(self, fps: int):
        """流处理循环"""
        last_process_time = time.time()
        process_interval = 1.0 / fps if fps > 0 else 0.1
        frame_id = 0
        
        try:
            while self.processing:
                frame_num, frame = self.stream_capture.read()
                if frame is None:
                    break
                
                current_time = time.time()
                if current_time - last_process_time >= process_interval:
                    frame_id += 1
                    last_process_time = current_time
                    
                    result = await self.process_frame(frame)
                    self.stats['frames_processed'] += 1
                    
                    # 输出 MCP 消息
                    if frame_id % 5 == 0:
                        self._print_mcp_message(result)
                
                time.sleep(0.01)
        
        finally:
            self.stream_capture.stop()
    
    def _print_mcp_message(self, msg: MCPMessage):
        """打印 MCP 消息"""
        logger.info(f"[MCP] {msg.msg_type}")
        logger.info(f"  ID: {msg.msg_id}")
        logger.info(f"  内容:\n{msg.content[:300]}")
        logger.info("-" * 50)
    
    def start(self):
        """启动服务"""
        self.processing = True
        logger.info("=" * 50)
        logger.info("Video LLM MCP 服务启动")
        logger.info(f"  YOLO模型: {self.detector.model_path}")
        logger.info(f"  LLM: {self.llm.provider}/{self.llm.model}")
        logger.info("=" * 50)
    
    def stop(self):
        """停止服务"""
        self.processing = False
        if self.stream_capture:
            self.stream_capture.stop()
        elapsed = time.time() - self.stats['start_time']
        logger.info("=" * 50)
        logger.info("服务统计:")
        logger.info(f"  运行时间: {elapsed:.1f}秒")
        logger.info(f"  处理帧数: {self.stats['frames_processed']}")
        logger.info(f"  检测数量: {self.stats['detections_count']}")
        logger.info(f"  LLM调用: {self.stats['llm_calls']}")
        logger.info(f"  MCP消息: {self.stats['mcp_messages']}")
        logger.info("=" * 50)


# ============ 主函数 ============
async def main():
    parser = argparse.ArgumentParser(description='视频流 + 大模型 + MCP 服务')
    
    # 运行模式
    parser.add_argument('--mode', type=str, default='camera',
                       choices=['camera', 'rtsp', 'file'],
                       help='运行模式')
    
    # 视频源
    parser.add_argument('--camera-id', type=int, default=0,
                       help='摄像头ID')
    parser.add_argument('--url', type=str, default=None,
                       help='RTSP流地址')
    parser.add_argument('--path', type=str, default=None,
                       help='视频文件路径')
    
    # FPS
    parser.add_argument('--fps', type=int, default=10,
                       help='处理帧率')
    
    # YOLO 配置
    parser.add_argument('--yolo-model', type=str, 
                       default='models/yolov8/train/weights/best.pt',
                       help='YOLO模型路径')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU阈值')
    
    # LLM 配置
    parser.add_argument('--llm-provider', type=str, default='openai',
                       choices=['openai', 'qwen', 'zhipu', 'mock'],
                       help='LLM提供商')
    parser.add_argument('--llm-model', type=str, default='gpt-4o',
                       help='LLM模型')
    parser.add_argument('--api-key', type=str, default='',
                       help='API Key')
    parser.add_argument('--llm-url', type=str, default=None,
                       help='API Base URL')
    
    # ASR 配置
    parser.add_argument('--asr-provider', type=str, default='mock',
                       help='ASR提供商')
    
    args = parser.parse_args()
    
    # 创建配置
    config = {
        'yolo_model': args.yolo_model,
        'conf_threshold': args.conf,
        'iou_threshold': args.iou,
        'llm_provider': args.llm_provider,
        'llm_model': args.llm_model,
        'llm_base_url': args.llm_url,
        'api_key': args.api_key or os.getenv('OPENAI_API_KEY', ''),
        'asr_provider': args.asr_provider
    }
    
    # 创建服务
    service = VideoLLMMCPService(config)
    service.start()
    
    try:
        if args.mode == 'camera':
            await service.run_camera_mode(args.camera_id, args.fps)
        elif args.mode == 'rtsp':
            url = args.url or 'rtsp://example.com/stream'
            await service.run_rtsp_mode(url, args.fps)
        elif args.mode == 'file':
            path = args.path or 'test.mp4'
            await service.run_file_mode(path, args.fps)
    finally:
        service.stop()
    
    logger.info(f"\n日志: {log_file}")


if __name__ == '__main__':
    asyncio.run(main())
