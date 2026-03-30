"""
PLC通信模块 - Modbus TCP
用于工控机与产线PLC对接
"""
import socket
import struct
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class PLCCommunicator:
    """PLC通信器 - Modbus TCP"""
    
    def __init__(self, ip: str = "192.168.1.100", port: int = 502):
        self.ip = ip
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.connected = False
        
    def connect(self) -> bool:
        """连接PLC"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)
            self.socket.connect((self.ip, self.port))
            self.connected = True
            logger.info(f"PLC连接成功: {self.ip}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"PLC连接失败: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """断开PLC"""
        if self.socket:
            self.socket.close()
            self.connected = False
            logger.info("PLC已断开")
    
    def write_register(self, address: int, value: int) -> bool:
        """写入保持寄存器 (Modbus功能码06)"""
        if not self.connected:
            if not self.connect():
                return False
        
        try:
            # Modbus TCP: 事务ID + 协议ID + 长度 + 功能码06 + 地址 + 值
            transaction_id = 1
            protocol_id = 0
            unit_id = 1
            
            # 请求: [事务2B][协议2B][长度2B][单元1B][功能1B][地址2B][值2B]
            request = struct.pack(">HHHBBHH",
                transaction_id, protocol_id, 7,  # 长度=6+1
                unit_id, 0x06, address, value
            )
            
            self.socket.send(request)
            response = self.socket.recv(12)
            
            if len(response) >= 12:
                return True
            return False
        except Exception as e:
            logger.error(f"写入寄存器失败: {e}")
            self.connected = False
            return False
    
    def read_register(self, address: int, count: int = 1) -> Optional[List[int]]:
        """读取保持寄存器 (Modbus功能码03)"""
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            transaction_id = 1
            protocol_id = 0
            unit_id = 1
            
            request = struct.pack(">HHHBBHH",
                transaction_id, protocol_id, 6,
                unit_id, 0x03, address, count
            )
            
            self.socket.send(request)
            response = self.socket.recv(9 + count * 2)
            
            if len(response) >= 9:
                values = struct.unpack(f">{count}H", response[9:])
                return list(values)
            return None
        except Exception as e:
            logger.error(f"读取寄存器失败: {e}")
            self.connected = False
            return None


class DetectionPLC:
    """检测结果PLC输出"""
    
    def __init__(self, plc_ip: str = "192.168.1.100"):
        self.plc = PLCCommunicator(plc_ip)
        
    def send_result(self, result: Dict) -> bool:
        """
        发送检测结果到PLC
        
        PLC寄存器映射:
        - 4000: 检测状态 (0=空闲, 1=检测中, 2=完成)
        - 4001: 缺陷数量
        - 4002: 缺陷类型1
        - 4003: 缺陷置信度1 (0-1000, 实际/10)
        - ...
        """
        if not result.get("success", False):
            # 无缺陷
            self.plc.write_register(4000, 2)  # 完成
            self.plc.write_register(4001, 0)  # 无缺陷
            return True
        
        detections = result.get("detections", [])
        
        # 写入结果
        self.plc.write_register(4000, 2)      # 完成
        self.plc.write_register(4001, len(detections))  # 缺陷数量
        
        # 写入每个缺陷
        defect_codes = {
            "missing_hole": 1,
            "mouse_bite": 2,
            "open_circuit": 3,
            "short": 4,
            "spur": 5,
            "spurious_copper": 6
        }
        
        for i, det in enumerate(detections[:8]):  # 最多8个
            addr = 4002 + i * 2
            cls = det.get("class", "unknown")
            code = defect_codes.get(cls, 0)
            conf = int(det.get("confidence", 0) * 1000)
            
            self.plc.write_register(addr, code)      # 类型
            self.plc.write_register(addr + 1, conf)  # 置信度
        
        return True
    
    def receive_trigger(self) -> bool:
        """
        接收PLC触发信号
        
        触发信号地址: 4000 (轮询)
        """
        values = self.plc.read_register(4000, 1)
        if values and values[0] == 1:
            # 收到触发信号，复位
            self.plc.write_register(4000, 0)
            return True
        return False


# 使用示例
if __name__ == "__main__":
    plc = DetectionPLC("192.168.1.100")
    
    # 连接PLC
    if plc.plc.connect():
        print("PLC连接成功")
        
        # 模拟检测结果
        test_result = {
            "success": True,
            "detections": [
                {"class": "missing_hole", "confidence": 0.85},
                {"class": "short", "confidence": 0.72}
            ]
        }
        
        # 发送结果
        plc.send_result(test_result)
        print("结果已发送到PLC")
