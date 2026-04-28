"""
完整语音对话系统 - 集成视频流 + ASR + LLM + TTS

架构:
    摄像头/视频 -> YOLO检测 -> DeepSeek LLM -> TTS语音输出
                    |
    麦克风 -> ASR语音识别 -> 知识库搜索
                         |
                    MCP协议输出

使用方式:
    python run_voice_chat.py
    python run_voice_chat.py --camera 0 --tts edge --asr whisper
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path

# 设置工作目录
os.chdir(Path(__file__).parent)

# ============ 日志配置 ============
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('VoiceChat')


class VoiceChatSystem:
    """完整语音对话系统"""
    
    def __init__(self, config: dict):
        self.config = config
        self.running = False
        
        # 初始化组件
        self._init_components()
    
    def _init_components(self):
        """初始化组件"""
        logger.info("=" * 60)
        logger.info("初始化语音对话系统...")
        logger.info("=" * 60)
        
        # 1. 音频录制
        try:
            from av_streaming import AudioRecorder
            self.audio_recorder = AudioRecorder(
                sample_rate=self.config.get('sample_rate', 16000)
            )
            logger.info("✓ 音频录制模块")
        except Exception as e:
            logger.warning(f"音频录制不可用: {e}")
            self.audio_recorder = None
        
        # 2. 音频播放
        try:
            from av_streaming import AudioPlayer
            self.audio_player = AudioPlayer()
            logger.info("✓ 音频播放模块")
        except Exception as e:
            logger.warning(f"音频播放不可用: {e}")
            self.audio_player = None
        
        # 3. TTS 语音合成
        try:
            from voice_chat_cli import TTSEngine
            self.tts = TTSEngine(
                provider=self.config.get('tts_provider', 'edge'),
                voice=self.config.get('tts_voice', 'zh-CN-XiaoxiaoNeural')
            )
            logger.info(f"✓ TTS 模块 ({self.config.get('tts_provider', 'edge')})")
        except Exception as e:
            logger.warning(f"TTS 不可用: {e}")
            self.tts = None
        
        # 4. ASR 语音识别
        try:
            from voice_chat_cli import ASREngine
            self.asr = ASREngine(
                provider=self.config.get('asr_provider', 'mock'),
                model_path=self.config.get('vosk_model', 'models/vosk')
            )
            logger.info(f"✓ ASR 模块 ({self.config.get('asr_provider', 'mock')})")
        except Exception as e:
            logger.warning(f"ASR 不可用: {e}")
            self.asr = None
        
        # 5. LLM 对话
        try:
            from voice_chat_cli import LLMInterface
            self.llm = LLMInterface(
                provider=self.config.get('llm_provider', 'deepseek'),
                model=self.config.get('llm_model', 'deepseek-chat'),
                api_key=self.config.get('api_key') or os.getenv('DEEPSEEK_API_KEY', '')
            )
            logger.info(f"✓ LLM 模块 ({self.config.get('llm_provider', 'deepseek')})")
        except Exception as e:
            logger.warning(f"LLM 不可用: {e}")
            self.llm = None
        
        # 6. 知识库搜索
        try:
            from voice_chat_cli import KnowledgeBaseSearch
            self.kb = KnowledgeBaseSearch(
                base_url=self.config.get('kb_url', 'http://localhost:8080')
            )
            logger.info("✓ 知识库模块")
        except Exception as e:
            logger.warning(f"知识库不可用: {e}")
            self.kb = None
        
        # 对话历史
        self.conversation_history = [
            {"role": "system", "content": """你是一个专业的PCB缺陷检测助手，名字叫小智。
你可以分析PCB图像中的缺陷，如缺孔(missing_hole)、鼠咬(mouse_bite)、开路(open_circuit)、短路(short)、毛刺(spur)、铜渣(copper)等。
请用简洁专业的语言回答，适合语音播报。"""}
        ]
        
        logger.info("=" * 60)
    
    async def start(self):
        """启动系统"""
        self.running = True
        
        logger.info("🎤 语音对话系统已启动")
        logger.info("=" * 60)
        logger.info("  按键说明:")
        logger.info("    Q - 退出")
        logger.info("    R - 开始语音对话 (录音)")
        logger.info("    D - 分析当前画面")
        logger.info("    P - 播放最后回复")
        logger.info("=" * 60)
        
        try:
            import cv2
            
            # 打开摄像头
            cap = None
            camera_id = self.config.get('camera_id', 0)
            
            try:
                cap = cv2.VideoCapture(camera_id)
                if cap.isOpened():
                    logger.info(f"✓ 摄像头已打开 (ID: {camera_id})")
                else:
                    logger.warning("无法打开摄像头，使用图片模式")
            except Exception as e:
                logger.warning(f"摄像头初始化失败: {e}")
            
            while self.running:
                if cap and cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        # 显示画面
                        cv2.imshow('Voice Chat - Q:quit R:record D:detect P:play', frame)
                        key = cv2.waitKey(1) & 0xFF
                        
                        if key == ord('q'):
                            break
                        elif key == ord('r'):  # R - 语音对话
                            await self.voice_dialog(frame)
                        elif key == ord('d'):  # D - 图像分析
                            await self.analyze_frame(frame)
                        elif key == ord('p'):  # P - 播放
                            self.play_last_response()
                else:
                    await asyncio.sleep(1)
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        await self.voice_dialog(None)
                    elif key == ord('d'):
                        # 使用测试图片
                        frame = cv2.imread('yolo_pcb_dataset/images/test/01_missing_hole_02.jpg')
                        if frame is not None:
                            await self.analyze_frame(frame)
                    elif key == ord('p'):
                        self.play_last_response()
        
        except KeyboardInterrupt:
            logger.info("收到中断信号")
        finally:
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            self.running = False
    
    async def voice_dialog(self, frame):
        """语音对话"""
        logger.info("-" * 40)
        logger.info("🎤 开始语音对话...")
        
        # 1. 录音
        transcript = await self._record_and_recognize()
        
        if not transcript:
            logger.info("未识别到语音")
            return
        
        print(f"\n👤 你说: {transcript}")
        
        # 2. 添加到对话历史
        self.conversation_history.append({
            "role": "user",
            "content": transcript
        })
        
        # 3. 调用 LLM
        response = await self._call_llm(frame, transcript)
        
        if response:
            # 4. 保存回复
            self.last_response = response
            self.last_audio = None
            
            print(f"\n🤖 助手: {response[:200]}...")
            
            # 5. TTS 语音输出
            await self._speak(response)
    
    async def analyze_frame(self, frame):
        """分析视频帧"""
        logger.info("-" * 40)
        logger.info("📷 分析画面...")
        
        # YOLO 检测
        if self.kb:
            import cv2
            import base64
            
            _, buffer = cv2.imencode('.jpg', frame)
            img_bytes = buffer.tobytes()
            
            detection = self.kb.detect_image(img_bytes)
            detection_text = self._format_detection(detection)
            
            logger.info(f"检测结果: {detection_text}")
            
            # 保存带标注的图像
            result_path = "output/detection_result.jpg"
            cv2.imwrite(result_path, frame)
            logger.info(f"✓ 图像已保存: {result_path}")
            
            # 调用 LLM 分析
            self.conversation_history.append({
                "role": "user",
                "content": f"请分析这张PCB图像: {detection_text}"
            })
            
            response = await self._call_llm(frame, detection_text)
            
            if response:
                self.last_response = response
                print(f"\n🤖 助手: {response[:200]}...")
                await self._speak(response)
    
    async def _record_and_recognize(self) -> str:
        """录音并识别"""
        if not self.audio_recorder:
            # 模拟模式
            await asyncio.sleep(2)
            return input("请输入你说的话: ").strip() or "请分析一下这个PCB板子"
        
        try:
            self.audio_recorder.start()
            logger.info("录音中 (5秒)...")
            await asyncio.sleep(5)
            audio_data = self.audio_recorder.stop()
            
            if self.asr:
                transcript = self.asr.recognize(audio_data)
                return transcript
            
            return "[模拟识别]"
        except Exception as e:
            logger.error(f"录音失败: {e}")
            return ""
    
    async def _call_llm(self, frame, context: str) -> str:
        """调用 LLM"""
        if not self.llm:
            return f"[模拟回复] 根据画面分析: 发现可能的缺陷，建议进一步检查。"
        
        try:
            response = self.llm.chat(self.conversation_history)
            
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            
            return response
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return f"[错误] {str(e)}"
    
    async def _speak(self, text: str):
        """语音播报"""
        if not self.tts:
            logger.info(f"[TTS] {text[:50]}...")
            return
        
        try:
            # 清理文本
            clean_text = text[:500].replace('**', '').replace('*', '').replace('#', '')
            
            audio_file = await self.tts.speak(clean_text)
            
            if audio_file and self.audio_player:
                logger.info(f"🎵 播放: {audio_file}")
                self.last_audio = audio_file
                self.audio_player.play_file(audio_file)
        except Exception as e:
            logger.error(f"TTS 失败: {e}")
    
    def play_last_response(self):
        """播放最后回复"""
        if hasattr(self, 'last_audio') and self.last_audio:
            if self.audio_player:
                self.audio_player.play_file(self.last_audio)
        else:
            logger.info("没有可播放的音频")
    
    def _format_detection(self, detection: dict) -> str:
        """格式化检测结果"""
        detections = detection.get('detections', [])
        
        if not detections:
            return "未检测到明显缺陷"
        
        lines = [f"检测到 {len(detections)} 个目标:"]
        for det in detections[:5]:
            lines.append(
                f"- {det.get('class_name', 'unknown')}: "
                f"置信度 {det.get('confidence', 0):.1%}"
            )
        
        return "; ".join(lines)


def main():
    parser = argparse.ArgumentParser(description='语音对话系统')
    
    # 摄像头
    parser.add_argument('--camera', type=int, default=0,
                       help='摄像头ID')
    
    # LLM
    parser.add_argument('--llm', type=str, default='deepseek',
                       choices=['deepseek', 'ollama', 'openai', 'mock'],
                       help='LLM提供商')
    parser.add_argument('--model', type=str, default='deepseek-chat',
                       help='LLM模型')
    parser.add_argument('--api-key', type=str, default='',
                       help='API Key')
    
    # TTS
    parser.add_argument('--tts', type=str, default='edge',
                       choices=['edge', 'gtts', 'mock'],
                       help='TTS提供商')
    parser.add_argument('--voice', type=str, default='zh-CN-XiaoxiaoNeural',
                       help='TTS声音')
    
    # ASR
    parser.add_argument('--asr', type=str, default='mock',
                       choices=['whisper', 'vosk', 'mock'],
                       help='ASR提供商')
    parser.add_argument('--vosk-model', type=str, default='models/vosk',
                       help='Vosk模型路径')
    
    # 其他
    parser.add_argument('--kb-url', type=str, default='http://localhost:8080',
                       help='知识库服务')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='采样率')
    
    args = parser.parse_args()
    
    # 创建配置
    config = {
        'camera_id': args.camera,
        'llm_provider': args.llm,
        'llm_model': args.model,
        'api_key': args.api_key,
        'tts_provider': args.tts,
        'tts_voice': args.voice,
        'asr_provider': args.asr,
        'vosk_model': args.vosk_model,
        'kb_url': args.kb_url,
        'sample_rate': args.sample_rate,
    }
    
    # 启动系统
    system = VoiceChatSystem(config)
    
    try:
        asyncio.run(system.start())
    except KeyboardInterrupt:
        logger.info("系统退出")


if __name__ == '__main__':
    main()
