"""
视频流 + 语音对话 CLI 客户端

功能:
    - 摄像头视频预览
    - 麦克风录音
    - 实时语音识别 (ASR)
    - 传输视频流 + 音频到服务器
    - 接收大模型回复并语音合成 (TTS) 输出

使用方式:
    python voice_chat_cli.py --mode camera
    python voice_chat_cli.py --mode rtsp --url rtsp://xxx
"""

import os
import sys
import time
import asyncio
import json
import base64
import queue
import threading
import argparse
import logging
import wave
import struct
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import cv2
import numpy as np

# 音视频模块
try:
    from av_streaming import (
        AudioRecorder, AudioPlayer, VideoCapture, 
        AudioCodec, VideoCodec, AVSynchronizer,
        RTMPStreamer, WebRTCStreamer
    )
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False
    logger.warning("av_streaming 模块不可用")

# ============ 日志配置 ============
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('VoiceChatCLI')


# ============ 音频采集 ============
class AudioCapturer:
    """麦克风音频采集"""
    
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.recording = False
        self.audio_thread = None
        self.audio_queue = queue.Queue()
        self._pyaudio = None
        self._stream = None
        self._init_audio()
    
    def _init_audio(self):
        """初始化音频设备"""
        try:
            import pyaudio
            self._pyaudio = pyaudio
            logger.info(f"✓ PyAudio 初始化 (采样率: {self.sample_rate})")
        except ImportError:
            logger.warning("PyAudio 未安装，使用模拟模式")
            self._pyaudio = None
    
    def start(self):
        """开始录音"""
        if self._pyaudio is None:
            logger.info("[模拟] 音频采集开始")
            self.recording = True
            return
        
        try:
            self._pyaudio = pyaudio.PyAudio()
            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            self.recording = True
            
            self.audio_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.audio_thread.start()
            
            logger.info("✓ 麦克风录音开始")
        except Exception as e:
            logger.error(f"音频设备打开失败: {e}")
            self.recording = False
    
    def _capture_loop(self):
        """采集循环"""
        frames = []
        
        while self.recording:
            try:
                data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)
                
                # 每2秒打包发送
                if len(frames) >= (self.sample_rate // self.chunk_size) * 2:
                    audio_data = b''.join(frames)
                    self.audio_queue.put(audio_data)
                    frames = []
                    
            except Exception as e:
                logger.error(f"音频采集错误: {e}")
                break
        
        # 发送剩余数据
        if frames:
            audio_data = b''.join(frames)
            self.audio_queue.put(audio_data)
    
    def read(self) -> Optional[bytes]:
        """读取音频数据"""
        try:
            return self.audio_queue.get(timeout=0.1)
        except queue.Empty:
            return None
    
    def stop(self):
        """停止录音"""
        self.recording = False
        
        try:
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
            
            if self._pyaudio and hasattr(self._pyaudio, 'PyAudio'):
                self._pyaudio.PyAudio().terminate()
            elif self._pyaudio and hasattr(self._pyaudio, 'terminate'):
                self._pyaudio.terminate()
        except Exception as e:
            logger.warning(f"Audio stop warning: {e}")
        
        logger.info("麦克风录音停止")


# ============ TTS 语音合成 ============
class TTSEngine:
    """语音合成引擎"""
    
    def __init__(self, provider: str = 'edge', voice: str = 'zh-CN-XiaoxiaoNeural'):
        self.provider = provider
        self.voice = voice
        self._edge_tts = None
        self._edge_voices = None
        self._init_tts()
    
    def _init_tts(self):
        """初始化 TTS"""
        if self.provider == 'edge':
            try:
                import edge_tts
                self._edge_tts = edge_tts
                logger.info(f"✓ Edge TTS 初始化 (Voice: {self.voice})")
            except ImportError:
                logger.warning("Edge TTS 未安装，使用模拟模式")
        elif self.provider == 'gtts':
            try:
                from gtts import gTTS
                self._gtts = gTTS
                logger.info("✓ gTTS 初始化")
            except ImportError:
                logger.warning("gTTS 未安装")
    
    async def speak(self, text: str, output_file: str = None) -> Optional[str]:
        """合成语音"""
        if not text:
            return None
        
        if self._edge_tts:
            return await self._edge_tts_speak(text, output_file)
        elif hasattr(self, '_gtts'):
            return self._gtts_speak(text, output_file)
        else:
            logger.info(f"[TTS] {text}")
            return None
    
    async def _edge_tts_speak(self, text: str, output_file: str = None) -> Optional[str]:
        """Edge TTS 合成"""
        try:
            if output_file is None:
                output_file = f"output/tts_{int(time.time())}.mp3"
            
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            communicate = self._edge_tts.Communicate(text, self.voice)
            await communicate.save(output_file)
            
            logger.info(f"✓ 语音已保存: {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"TTS 失败: {e}")
            return None
    
    def _gtts_speak(self, text: str, output_file: str = None) -> Optional[str]:
        """gTTS 合成"""
        try:
            if output_file is None:
                output_file = f"output/tts_{int(time.time())}.mp3"
            
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            tts = self._gtts(text=text, lang='zh')
            tts.save(output_file)
            
            logger.info(f"✓ 语音已保存: {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"gTTS 失败: {e}")
            return None
    
    async def speak_and_play(self, text: str):
        """合成并播放"""
        audio_file = await self.speak(text)
        if audio_file and os.path.exists(audio_file):
            self._play_audio(audio_file)
    
    def _play_audio(self, audio_file: str):
        """播放音频文件"""
        try:
            if sys.platform == 'win32':
                import winsound
                winsound.PlaySound(audio_file, winsound.SND_FILENAME)
            else:
                os.system(f"afplay '{audio_file}' 2>/dev/null || mpg123 '{audio_file}' 2>/dev/null")
        except Exception as e:
            logger.warning(f"播放失败: {e}")


# ============ ASR 语音识别 ============
class ASREngine:
    """语音识别引擎"""
    
    def __init__(self, provider: str = 'vosk', model_path: str = 'models/vosk'):
        self.provider = provider
        self.model_path = model_path
        self.model = None
        self.recognizer = None
        self._init_asr()
    
    def _init_asr(self):
        """初始化 ASR"""
        if self.provider == 'vosk':
            try:
                from vosk import Model, KaldiRecognizer
                if Path(self.model_path).exists():
                    self.model = Model(self.model_path)
                    self.recognizer = KaldiRecognizer(self.model, 16000)
                    logger.info(f"✓ Vosk ASR 初始化")
                else:
                    logger.warning("Vosk 模型不存在，使用模拟模式")
            except ImportError:
                logger.warning("Vosk 未安装")
        elif self.provider == 'whisper':
            try:
                import whisper
                self.model = whisper.load_model("base")
                logger.info("✓ Whisper ASR 初始化")
            except ImportError:
                logger.warning("Whisper 未安装")
        else:
            logger.info("ASR 使用模拟模式")
    
    def recognize(self, audio_data: bytes) -> str:
        """识别音频"""
        if self.recognizer:
            try:
                self.recognizer.AcceptWaveform(audio_data)
                result = json.loads(self.recognizer.Result())
                return result.get('text', '')
            except Exception as e:
                logger.error(f"ASR 识别失败: {e}")
        
        return "[模拟识别结果]"


# ============ WebSocket 通信 ============
class VoiceChatWebSocket:
    """语音聊天 WebSocket 客户端"""
    
    def __init__(self, server_url: str = 'ws://localhost:8765'):
        self.server_url = server_url
        self.ws = None
        self.connected = False
    
    async def connect(self) -> bool:
        """连接服务器"""
        try:
            import websockets
            async with websockets.connect(self.server_url) as ws:
                self.ws = ws
                self.connected = True
                logger.info(f"✓ 已连接: {self.server_url}")
                
                # 接收欢迎消息
                msg = await ws.recv()
                logger.info(f"服务器: {msg}")
                
                return True
        except Exception as e:
            logger.error(f"连接失败: {e}")
            return False
    
    async def send_audio(self, audio_data: bytes, metadata: Dict = None):
        """发送音频"""
        if not self.connected or not self.ws:
            return
        
        try:
            audio_base64 = base64.b64encode(audio_data).decode()
            message = {
                'type': 'voice_input',
                'timestamp': time.time(),
                'audio': audio_base64,
                'sample_rate': 16000,
                'metadata': metadata or {}
            }
            await self.ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"发送音频失败: {e}")
    
    async def send_video_frame(self, frame: np.ndarray, metadata: Dict = None):
        """发送视频帧"""
        if not self.connected or not self.ws:
            return
        
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            img_base64 = base64.b64encode(buffer).decode()
            
            message = {
                'type': 'video_frame',
                'timestamp': time.time(),
                'frame': img_base64,
                'metadata': metadata or {}
            }
            await self.ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"发送视频失败: {e}")
    
    async def receive(self) -> Optional[Dict]:
        """接收消息"""
        if not self.connected or not self.ws:
            return None
        
        try:
            msg = await asyncio.wait_for(self.ws.recv(), timeout=1.0)
            return json.loads(msg)
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"接收失败: {e}")
            return None
    
    async def close(self):
        """关闭连接"""
        self.connected = False
        if self.ws:
            await self.ws.close()


# ============ 知识库搜索 ============
class KnowledgeBaseSearch:
    """知识库搜索"""
    
    def __init__(self, base_url: str = 'http://localhost:8080'):
        self.base_url = base_url
        self.session = None
        self._init_session()
    
    def _init_session(self):
        """初始化会话"""
        try:
            import requests
            self.session = requests.Session()
            logger.info(f"✓ 知识库连接: {self.base_url}")
        except ImportError:
            logger.warning("requests 未安装")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索知识库"""
        if not self.session:
            return [{'content': f'[模拟] 关于 "{query}" 的回答', 'score': 0.95}]
        
        try:
            response = self.session.post(
                f"{self.base_url}/search",
                json={'query': query, 'top_k': top_k},
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get('results', [])
        except Exception as e:
            logger.error(f"搜索失败: {e}")
        
        return [{'content': f'[模拟] 关于 "{query}" 的回答', 'score': 0.95}]
    
    def detect_image(self, image_data: bytes) -> Dict:
        """检测图像"""
        if not self.session:
            return {'detections': [], 'count': 0}
        
        try:
            img_base64 = base64.b64encode(image_data).decode()
            response = self.session.post(
                f"{self.base_url}/detect",
                json={'data': img_base64, 'confidence': 0.25},
                timeout=60
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"检测失败: {e}")
        
        return {'detections': [], 'count': 0}


# ============ LLM 集成 ============
class LLMInterface:
    """LLM 接口 (支持 Ollama / DeepSeek / OpenAI)"""
    
    PROVIDERS = ['ollama', 'deepseek', 'openai', 'qwen', 'mock']
    
    def __init__(self, provider: str = 'ollama', model: str = None,
                 base_url: str = None, api_key: str = None):
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self._init_client()
    
    def _init_client(self):
        """初始化客户端"""
        if self.provider == 'ollama':
            self.base_url = self.base_url or 'http://localhost:11434'
            self.model = self.model or 'qwen2.5:7b'
            self._test_ollama()
        elif self.provider == 'deepseek':
            self.base_url = 'https://api.deepseek.com/v1'
            self.model = self.model or 'deepseek-chat'
            self.api_key = self.api_key or os.getenv('DEEPSEEK_API_KEY', '')
            logger.info(f"✓ DeepSeek: {self.model}")
        elif self.provider in ['openai', 'qwen']:
            self.base_url = self.base_url or 'https://api.openai.com/v1'
            self.api_key = self.api_key or os.getenv('OPENAI_API_KEY', '')
            logger.info(f"✓ {self.provider.title()}: {self.model}")
        else:
            logger.info("LLM: 模拟模式")
    
    def _test_ollama(self):
        """测试 Ollama 连接"""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                logger.info(f"✓ Ollama 连接成功: {[m['name'] for m in models]}")
        except Exception as e:
            logger.warning(f"Ollama 未运行: {e}")
    
    def chat(self, messages: List[Dict], stream: bool = False) -> str:
        """对话"""
        try:
            import requests
            
            if self.provider == 'ollama':
                return self._chat_ollama(messages, stream)
            elif self.provider == 'deepseek':
                return self._chat_deepseek(messages)
            elif self.provider in ['openai', 'qwen']:
                return self._chat_openai(messages)
        
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
        
        return f"[模拟回复] 收到消息处理中..."
    
    def _chat_ollama(self, messages: List[Dict], stream: bool) -> str:
        """Ollama 对话"""
        import requests
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={'model': self.model, 'messages': messages, 'stream': stream},
            timeout=120
        )
        if response.status_code == 200:
            return response.json().get('message', {}).get('content', '')
        return "[Ollama 错误]"
    
    def _chat_deepseek(self, messages: List[Dict]) -> str:
        """DeepSeek 对话"""
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "messages": messages, "max_tokens": 1000}
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers, json=payload, timeout=60
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return "[DeepSeek 错误]"
    
    def _chat_openai(self, messages: List[Dict]) -> str:
        """OpenAI/Qwen 对话"""
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "messages": messages}
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers, json=payload, timeout=60
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return "[API 错误]"
    
    def vision_chat(self, image_data: bytes, text: str) -> str:
        """视觉对话 (如果有支持视觉的模型)"""
        # 目前 DeepSeek/OpenAI 需要用多模态模型
        return self.chat([
            {"role": "user", "content": text + "\n\n[图片数据]"}
        ])


# ============ 主 CLI 客户端 ============
class VoiceChatCLI:
    """语音对话 CLI 客户端"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # 初始化组件
        self.audio = AudioCapturer(
            sample_rate=self.config.get('sample_rate', 16000)
        )
        
        self.tts = TTSEngine(
            provider=self.config.get('tts_provider', 'edge'),
            voice=self.config.get('tts_voice', 'zh-CN-XiaoxiaoNeural')
        )
        
        self.asr = ASREngine(
            provider=self.config.get('asr_provider', 'whisper'),
            model_path=self.config.get('vosk_model', 'models/vosk')
        )
        
        self.llm = LLMInterface(
            provider=self.config.get('llm_provider', 'deepseek'),
            model=self.config.get('llm_model', 'deepseek-chat'),
            base_url=self.config.get('llm_url'),
            api_key=self.config.get('api_key') or os.getenv('DEEPSEEK_API_KEY', '')
        )
        
        self.kb = KnowledgeBaseSearch(
            base_url=self.config.get('kb_url', 'http://localhost:8080')
        )
        
        self.running = False
        self.stats = {
            'audio_chunks': 0,
            'video_frames': 0,
            'llm_calls': 0,
            'start_time': time.time()
        }
        
        # 对话历史
        self.conversation_history = [
            {"role": "system", "content": "你是一个专业的PCB缺陷检测助手，请用简洁专业的语言回答。"}
        ]
    
    def start(self):
        """启动"""
        self.running = True
        logger.info("=" * 50)
        logger.info("语音对话 CLI 启动")
        logger.info(f"  TTS: {self.config.get('tts_provider', 'edge')}")
        logger.info(f"  ASR: {self.config.get('asr_provider', 'whisper')}")
        logger.info(f"  LLM: {self.llm.model}")
        logger.info("=" * 50)
    
    def stop(self):
        """停止"""
        self.running = False
        self.audio.stop()
        
        elapsed = time.time() - self.stats['start_time']
        logger.info("=" * 50)
        logger.info("统计:")
        logger.info(f"  运行时间: {elapsed:.1f}秒")
        logger.info(f"  音频块: {self.stats['audio_chunks']}")
        logger.info(f"  视频帧: {self.stats['video_frames']}")
        logger.info(f"  LLM调用: {self.stats['llm_calls']}")
        logger.info("=" * 50)
    
    async def run_camera_mode(self):
        """摄像头模式"""
        logger.info("启动摄像头模式...")
        
        # 打开摄像头
        cap = None
        try:
            cap = cv2.VideoCapture(self.config.get('camera_id', 0))
            if cap.isOpened():
                logger.info("摄像头已打开")
            else:
                logger.warning("无法打开摄像头，使用图片模式")
                cap = None
        except Exception as e:
            logger.warning(f"摄像头初始化失败: {e}")
            cap = None
        
        # 启动音频采集 (模拟模式)
        logger.info("音频使用模拟模式")
        self.audio.recording = True
        
        try:
            while self.running:
                # 读取视频帧
                frame = None
                if cap:
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        cv2.imshow('Voice Chat CLI - Q:quit, Space:detect, R:voice', frame)
                        key = cv2.waitKey(1) & 0xFF
                    else:
                        key = cv2.waitKey(100) & 0xFF
                else:
                    # 无摄像头，每2秒处理一次图片
                    await asyncio.sleep(2)
                    key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == key if 'key' in dir() else 0:
                    pass
                
                # 非阻塞处理音频队列
                audio_data = self.audio.read()
                if audio_data:
                    self.stats['audio_chunks'] += 1
        
        finally:
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            self.audio.stop()
    
    async def _process_frame_with_llm(self, frame: np.ndarray):
        """用 LLM 处理视频帧"""
        logger.info("正在分析图像...")
        
        # 编码图像
        _, buffer = cv2.imencode('.jpg', frame)
        img_bytes = buffer.tobytes()
        
        # 发送到知识库检测
        detection = self.kb.detect_image(img_bytes)
        
        # 构建 LLM 提示
        detection_text = self._format_detection(detection)
        
        self.conversation_history.append({
            "role": "user",
            "content": f"请分析这张PCB图像的检测结果: {detection_text}"
        })
        
        # 调用 LLM
        response = self.llm.chat(self.conversation_history)
        
        # 保存对话
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        self.stats['llm_calls'] += 1
        
        # 显示结果
        print("\n" + "=" * 60)
        print("🤖 AI 助手:")
        print("=" * 60)
        print(response)
        print("=" * 60 + "\n")
        
        # TTS 播报
        await self._speak_response(response)
    
    async def _voice_dialog(self):
        """语音对话"""
        logger.info("开始语音对话 (录音 5 秒)...")
        
        # 临时保存音频
        frames = []
        self.audio.start()
        
        start_time = time.time()
        while time.time() - start_time < 5:
            audio_data = self.audio.read()
            if audio_data:
                frames.append(audio_data)
        
        self.audio.stop()
        
        if not frames:
            logger.warning("未采集到音频")
            return
        
        audio_data = b''.join(frames)
        
        # ASR 识别
        transcript = self.asr.recognize(audio_data)
        
        print("\n" + "=" * 60)
        print("🎤 你说:")
        print("=" * 60)
        print(transcript)
        print("=" * 60 + "\n")
        
        if not transcript or len(transcript) < 2:
            return
        
        # 添加到对话历史
        self.conversation_history.append({
            "role": "user",
            "content": transcript
        })
        
        # 调用 LLM
        response = self.llm.chat(self.conversation_history)
        
        # 保存对话
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        self.stats['llm_calls'] += 1
        
        # 显示回复
        print("=" * 60)
        print("🤖 AI 助手:")
        print("=" * 60)
        print(response)
        print("=" * 60 + "\n")
        
        # TTS 播报
        await self._speak_response(response)
    
    async def _speak_response(self, text: str):
        """TTS 播报回复"""
        # 截取前500字
        short_text = text[:500] if len(text) > 500 else text
        
        # 简化文本用于 TTS
        clean_text = short_text.replace('**', '').replace('*', '').replace('#', '')
        
        audio_file = await self.tts.speak(clean_text)
        
        if audio_file and os.path.exists(audio_file):
            logger.info(f"✓ 语音已生成: {audio_file}")
            # 可以播放: self.tts._play_audio(audio_file)
    
    def _format_detection(self, detection: Dict) -> str:
        """格式化检测结果"""
        detections = detection.get('detections', [])
        
        if not detections:
            return "未检测到缺陷"
        
        lines = []
        for det in detections:
            lines.append(
                f"{det.get('class_name', 'unknown')}: "
                f"置信度 {det.get('confidence', 0):.1%}"
            )
        
        return "; ".join(lines)


# ============ 主函数 ============
async def main():
    parser = argparse.ArgumentParser(description='语音对话 CLI 客户端')
    
    # 模式
    parser.add_argument('--mode', type=str, default='camera',
                       choices=['camera', 'rtsp', 'file'],
                       help='运行模式')
    
    # 视频源
    parser.add_argument('--camera-id', type=int, default=0,
                       help='摄像头ID')
    parser.add_argument('--url', type=str, default=None,
                       help='RTSP流地址')
    parser.add_argument('--video', type=str, default=None,
                       help='视频文件')
    
    # 服务器
    parser.add_argument('--server', type=str, 
                       default='ws://localhost:8765',
                       help='WebSocket服务器')
    parser.add_argument('--kb-url', type=str,
                       default='http://localhost:8080',
                       help='知识库服务地址')
    parser.add_argument('--ollama-url', type=str,
                       default='http://localhost:11434',
                       help='Ollama服务地址')
    
    # LLM
    parser.add_argument('--llm-provider', type=str,
                       default='deepseek',
                       choices=['ollama', 'deepseek', 'openai', 'qwen', 'mock'],
                       help='LLM提供商')
    parser.add_argument('--llm-model', type=str,
                       default='deepseek-chat',
                       help='LLM模型')
    parser.add_argument('--llm-url', type=str,
                       default=None,
                       help='API URL (可选)')
    parser.add_argument('--api-key', type=str,
                       default='',
                       help='API Key')
    
    # TTS
    parser.add_argument('--tts-provider', type=str,
                       default='edge',
                       choices=['edge', 'gtts', 'mock'],
                       help='TTS提供商')
    parser.add_argument('--tts-voice', type=str,
                       default='zh-CN-XiaoxiaoNeural',
                       help='TTS声音')
    
    # ASR
    parser.add_argument('--asr-provider', type=str,
                       default='whisper',
                       choices=['whisper', 'vosk', 'mock'],
                       help='ASR提供商')
    parser.add_argument('--vosk-model', type=str,
                       default='models/vosk',
                       help='Vosk模型路径')
    
    args = parser.parse_args()
    
    # 创建配置
    config = {
        'sample_rate': 16000,
        'camera_id': args.camera_id,
        'server_url': args.server,
        'kb_url': args.kb_url,
        'llm_provider': args.llm_provider,
        'llm_model': args.llm_model,
        'llm_url': args.llm_url,
        'api_key': args.api_key,
        'tts_provider': args.tts_provider,
        'tts_voice': args.tts_voice,
        'asr_provider': args.asr_provider,
        'vosk_model': args.vosk_model
    }
    
    # 创建客户端
    client = VoiceChatCLI(config)
    client.start()
    
    try:
        await client.run_camera_mode()
    except KeyboardInterrupt:
        logger.info("收到中断信号")
    finally:
        client.stop()


if __name__ == '__main__':
    asyncio.run(main())
