"""
PortAI PLC 连接测试客户端
测试上位机与PLC模拟器的连接
"""
import socket
import time
import struct

def connect_to_plc(host='127.0.0.1', port=5000):
    """连接PLC并读取寄存器"""
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(5)
        client.connect((host, port))
        print(f"[OK] 已连接到 PLC {host}:{port}")
        
        reg_names = ['CRANE_ST', 'CONT_POS', 'TRUCK_CNT', 'SHIP_DET', 
                     'ALARM', 'SPEED', 'TEMP', 'WEIGHT']
        
        for i in range(5):  # 读取5次
            # Modbus TCP Read Holding Registers
            frame = bytes([
                0x00, 0x01,  # Transaction ID
                0x00, 0x00,  # Protocol ID
                0x00, 0x06,  # Length
                0x01,        # Unit ID
                0x03,        # Function: Read Holding Registers
                0x00, 0x00,  # Start Address
                0x00, 0x08   # Quantity: 8 registers
            ])
            
            client.send(frame)
            print(f"\n--- 第 {i+1} 次读取 ---")
            
            # 接收响应
            data = client.recv(256)
            
            if len(data) >= 9:
                byte_count = data[8]
                print(f"字节数: {byte_count}")
                
                for j in range(min(byte_count // 2, 8)):
                    value = struct.unpack('>H', data[9 + j*2 : 11 + j*2])[0]
                    print(f"  [{j}] {reg_names[j]}: {value}")
            
            time.sleep(1)
        
        client.close()
        print("\n[OK] 测试完成!")
        return True
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("PortAI PLC 连接测试")
    print("=" * 50)
    connect_to_plc()
    input("\n按回车键退出...")
