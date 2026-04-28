#!/usr/bin/env python3
"""
PortAI PLC模拟器 v2.0
支持 Modbus TCP 协议
修复了响应格式问题
"""
import socket
import struct
import threading
import time
import random


class PLCSimulator:
    def __init__(self, port=5001):  # 改为5001避免端口冲突
        self.running = True
        self.tick = 0
        self.port = port
        
        # 保持寄存器
        self.holding_registers = {
            0: 0,   # CRANE_ST
            1: 0,   # CONT_POS
            2: 0,   # TRUCK_CNT
            3: 0,   # SHIP_DET
            4: 0,   # ALARM
            5: 0,   # SPEED
            6: 25,  # TEMP
            7: 5000 # WEIGHT
        }
        
        # 输入寄存器
        self.input_registers = {
            0: 100,
            1: 200
        }
        
    def update_registers(self):
        """更新寄存器值"""
        self.tick += 1
        
        self.holding_registers[0] = random.randint(0, 2)    # crane_status
        self.holding_registers[1] = self.tick % 100        # container_pos
        self.holding_registers[2] = random.randint(0, 5)   # truck_count
        self.holding_registers[3] = random.randint(0, 1)  # ship_detected
        self.holding_registers[4] = 0                      # alarm
        self.holding_registers[5] = random.randint(0, 100) # speed
        self.holding_registers[6] = random.randint(20, 35) # temperature
        self.holding_registers[7] = random.randint(1000, 25000) # weight
        
    def handle_modbus_tcp(self, data):
        """处理 Modbus TCP 请求"""
        if len(data) < 12:
            return None
            
        try:
            # 解析 MBAP 头
            transaction_id = struct.unpack('>H', bytes(data[0:2]))[0]
            protocol_id = struct.unpack('>H', bytes(data[2:4]))[0]
            length = struct.unpack('>H', bytes(data[4:6]))[0]
            
            # 解析 PDU
            unit_id = data[6]
            function_code = data[7]
            
            print(f"[PLC] RX: {data.hex().upper()}")
            
            if function_code == 0x03:  # 读保持寄存器
                return self.read_holding_registers(data, transaction_id)
                
            elif function_code == 0x04:  # 读输入寄存器
                return self.read_input_registers(data, transaction_id)
                
            elif function_code == 0x06:  # 写单个寄存器
                return self.write_single_register(data, transaction_id)
                
            else:
                print(f"[PLC] 不支持的Function Code: {function_code:#x}")
                return self.error_response(data, transaction_id, function_code, 0x01)
                
        except Exception as e:
            print(f"[PLC] 解析错误: {e}")
            return None
            
    def read_holding_registers(self, data, tid):
        """读保持寄存器 (Function 0x03)"""
        try:
            start_addr = struct.unpack('>H', bytes(data[8:10]))[0]
            quantity = struct.unpack('>H', bytes(data[10:12]))[0]
            
            print(f"[PLC] Read Holding: addr={start_addr}, count={quantity}")
            
            # 构建响应
            response = struct.pack('>HH', tid, 0)  # Transaction ID, Protocol
            response += struct.pack('>H', 3 + quantity * 2)  # Length
            response += bytes([0x01, 0x03, quantity * 2])  # Unit ID, Function, Byte Count
            
            # 添加寄存器值
            for i in range(quantity):
                addr = start_addr + i
                value = self.holding_registers.get(addr, 0)
                response += struct.pack('>H', value)
                
            print(f"[PLC] TX: {response.hex().upper()}")
            return response
            
        except Exception as e:
            print(f"[PLC] 读寄存器错误: {e}")
            return None
            
    def read_input_registers(self, data, tid):
        """读输入寄存器 (Function 0x04)"""
        try:
            start_addr = struct.unpack('>H', bytes(data[8:10]))[0]
            quantity = struct.unpack('>H', bytes(data[10:12]))[0]
            
            response = struct.pack('>HH', tid, 0)
            response += struct.pack('>H', 3 + quantity * 2)
            response += bytes([0x01, 0x04, quantity * 2])
            
            for i in range(quantity):
                addr = start_addr + i
                value = self.input_registers.get(addr, 0)
                response += struct.pack('>H', value)
                
            return response
            
        except Exception as e:
            print(f"[PLC] 读输入寄存器错误: {e}")
            return None
            
    def write_single_register(self, data, tid):
        """写单个寄存器 (Function 0x06)"""
        try:
            addr = struct.unpack('>H', bytes(data[8:10]))[0]
            value = struct.unpack('>H', bytes(data[10:12]))[0]
            
            print(f"[PLC] Write: addr={addr}, value={value}")
            
            self.holding_registers[addr] = value
            
            # 原样返回 (成功)
            return bytes(data)
            
        except Exception as e:
            print(f"[PLC] 写寄存器错误: {e}")
            return None
            
    def error_response(self, data, tid, func_code, error_code):
        """错误响应"""
        response = struct.pack('>HH', tid, 0)
        response += bytes([0x04])  # Length
        response += bytes([0x01, func_code | 0x80, error_code])
        return response
        
    def run_tcp(self, host='127.0.0.1', port=None):
        """运行 TCP 服务器"""
        if port is None:
            port = self.port  # 使用初始化时的端口
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server.bind((host, port))
            server.listen(5)
            print(f"[PLC] Server started on {host}:{port}")
            print(f"[PLC] Waiting for connections...")
            
            while self.running:
                try:
                    client, addr = server.accept()
                    print(f"[PLC] Client connected: {addr}")
                    
                    # 更新寄存器
                    self.update_registers()
                    
                    # 处理客户端
                    client.settimeout(5.0)
                    
                    while self.running:
                        try:
                            data = client.recv(1024)
                            if not data:
                                break
                                
                            # 处理请求
                            response = self.handle_modbus_tcp(bytes(data))
                            
                            if response:
                                client.send(response)
                                
                            # 定期更新寄存器
                            self.update_registers()
                            
                        except socket.timeout:
                            self.update_registers()
                            continue
                        except Exception as e:
                            print(f"[PLC] 处理错误: {e}")
                            break
                            
                    client.close()
                    print(f"[PLC] Client disconnected: {addr}")
                    
                except Exception as e:
                    if self.running:
                        print(f"[PLC] Accept error: {e}")
                        
        except OSError as e:
            print(f"[PLC] Port binding failed: {e}")
        finally:
            server.close()
            print("[PLC] Server stopped")


def main():
    print("\n" + "="*50)
    print("  PortAI PLC Simulator v2.0")
    print("  Modbus TCP Server")
    print("="*50)
    print()
    print(f"  Port: 5001")
    print("  Protocol: Modbus TCP")
    print("  Registers: 0-7 (Holding)")
    print()
    print("="*50 + "\n")
    
    plc = PLCSimulator()
    
    # 启动TCP服务器
    thread = threading.Thread(target=plc.run_tcp, daemon=True)
    thread.start()
    
    print("PLC Simulator running...")
    print("Open HMI and connect to 127.0.0.1:5000")
    print("Press Ctrl+C to stop\n")
    
    # 显示状态
    try:
        while True:
            time.sleep(3)
            print(f"[Status] Tick: {plc.tick} | Registers: {[plc.holding_registers[i] for i in range(8)]}")
    except KeyboardInterrupt:
        print("\nStopping...")
        plc.running = False


if __name__ == "__main__":
    main()
