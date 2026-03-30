"""
MQTT 通信模块 - 发布/订阅检测结果
"""
import json
import paho.mqtt.client as mqtt
from typing import Dict, Any


class MQTTSender:
    """MQTT 发送端"""
    
    def __init__(self, broker: str = "192.168.1.10", port: int = 1883):
        self.broker = broker
        self.port = port
        # 使用新版API避免警告
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.connected = False
    
    def connect(self) -> bool:
        """连接MQTT broker"""
        try:
            self.client.connect(self.broker, self.port, 60)
            self.connected = True
            print(f"MQTT连接成功: {self.broker}:{self.port}")
            return True
        except Exception as e:
            print(f"MQTT连接失败: {e}")
            return False
    
    def publish(self, topic: str, payload: Dict) -> bool:
        """发布消息"""
        if not self.connected:
            self.connect()
        
        try:
            result = self.client.publish(topic, json.dumps(payload))
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"发布成功: {topic}")
                return True
            else:
                print(f"发布失败: {result.rc}")
                return False
        except Exception as e:
            print(f"发布失败: {e}")
            return False
    
    def publish_detection_result(self, result: Dict):
        """发布检测结果"""
        return self.publish("pcb/detection/result", result)
    
    def loop_start(self):
        """启动后台循环"""
        self.client.loop_start()


class MQTTReceiver:
    """MQTT 接收端 - 订阅触发信号"""
    
    def __init__(self, broker: str = "192.168.1.10", port: int = 1883):
        self.broker = broker
        self.port = port
        self.client = mqtt.Client()
        self.trigger_callback = None
        
        # 绑定回调
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
    
    def _on_connect(self, client, userdata, flags, rc):
        """连接回调"""
        if rc == 0:
            print(f"MQTT连接成功: {self.broker}")
            # 订阅触发主题
            client.subscribe("pcb/detection/trigger")
        else:
            print(f"MQTT连接失败: {rc}")
    
    def _on_message(self, client, userdata, msg):
        """消息回调"""
        try:
            payload = json.loads(msg.payload.decode())
            print(f"收到消息: {msg.topic} -> {payload}")
            
            if self.trigger_callback:
                self.trigger_callback(payload)
        except Exception as e:
            print(f"解析消息失败: {e}")
    
    def set_trigger_callback(self, callback):
        """设置触发回调"""
        self.trigger_callback = callback
    
    def connect(self) -> bool:
        """连接MQTT broker"""
        try:
            self.client.connect(self.broker, self.port, 60)
            return True
        except Exception as e:
            print(f"MQTT连接失败: {e}")
            return False
    
    def loop_forever(self):
        """阻塞运行"""
        self.client.loop_forever()


# 使用示例
if __name__ == "__main__":
    import time
    
    # 使用公共MQTT测试 broker
    print("=" * 50)
    print("MQTT 测试 - 使用公共 broker.emqx.io")
    print("=" * 50)
    
    sender = MQTTSender("broker.emqx.io", 1883)
    sender.connect()
    
    # 模拟检测结果
    test_result = {
        "success": True,
        "image_id": "pcb_001",
        "timestamp": "2026-03-30T22:40:00",
        "detections": [
            {"class": "missing_hole", "confidence": 0.85, "bbox": [100, 100, 200, 200]},
            {"class": "short", "confidence": 0.72, "bbox": [300, 300, 400, 400]}
        ],
        "total_defects": 2
    }
    
    # 发布
    sender.publish_detection_result(test_result)
    
    print("发布完成")
