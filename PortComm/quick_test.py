"""快速测试PLC连接"""
import socket
import struct
import time

def test_plc():
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(3)
        client.connect(('127.0.0.1', 5000))
        print("[OK] 连接成功!")
        
        # 发送Modbus读请求
        frame = bytes([
            0x00, 0x01, 0x00, 0x00, 0x00, 0x06,  # Header
            0x01, 0x03, 0x00, 0x00, 0x00, 0x08  # Read 8 regs
        ])
        
        for i in range(3):
            client.send(frame)
            data = client.recv(256)
            print(f"[RECV] {data.hex().upper()}")
            time.sleep(0.5)
            
        client.close()
        print("[OK] 测试完成!")
        
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    test_plc()
    input("按回车退出...")
