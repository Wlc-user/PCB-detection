"""
PortAI 工业通信系统 - 配置文件
真实PLC连接配置
"""
import json
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "config.json"

# 默认配置
DEFAULT_CONFIG = {
    # TCP连接配置 (模拟器: 127.0.0.1:5000, 真实PLC: 实际IP:502)
    "tcp": {
        "enabled": True,
        "host": "127.0.0.1",  # 模拟器用127.0.0.1, 真实PLC用实际IP
        "port": 5001,         # 模拟器用5001, 真实PLC用502
        "timeout": 5,
        "retry": 3
    },
    
    # 串口连接配置
    "serial": {
        "enabled": False,
        "port": "COM3",          # 真实串口号
        "baudrate": 9600,
        "bytesize": 8,
        "parity": "N",
        "stopbits": 1,
        "timeout": 1
    },
    
    # Modbus配置
    "modbus": {
        "slave_id": 1,
        "start_address": 0,
        "register_count": 8
    },
    
    # 数据寄存器映射
    "registers": {
        "0": {"name": "CRANE_ST", "desc": "起重机状态", "unit": ""},
        "1": {"name": "CONT_POS", "desc": "集装箱位置", "unit": ""},
        "2": {"name": "TRUCK_CNT", "desc": "卡车数量", "unit": "辆"},
        "3": {"name": "SHIP_DET", "desc": "船舶检测", "unit": ""},
        "4": {"name": "ALARM", "desc": "告警标志", "unit": ""},
        "5": {"name": "SPEED", "desc": "速度", "unit": "km/h"},
        "6": {"name": "TEMP", "desc": "温度", "unit": "°C"},
        "7": {"name": "WEIGHT", "desc": "重量", "unit": "kg"}
    },
    
    # 报警阈值
    "alarms": {
        "temperature_high": 35,
        "temperature_low": 10,
        "weight_max": 50000,
        "speed_max": 120
    },
    
    # 轮询间隔(ms)
    "poll_interval": 500,
    
    # 数据记录
    "data_log": {
        "enabled": True,
        "path": "data_log.csv",
        "interval": 5  # 秒
    }
}

def load_config():
    """加载配置"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"[OK] 加载配置: {CONFIG_FILE}")
            return config
        except Exception as e:
            print(f"[WARN] 配置加载失败: {e}")
    
    # 创建默认配置
    save_config(DEFAULT_CONFIG)
    return DEFAULT_CONFIG

def save_config(config):
    """保存配置"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"[OK] 保存配置: {CONFIG_FILE}")
        return True
    except Exception as e:
        print(f"[ERROR] 保存配置失败: {e}")
        return False

def get_connection_string(config):
    """获取连接字符串"""
    if config["tcp"]["enabled"]:
        return f"TCP://{config['tcp']['host']}:{config['tcp']['port']}"
    elif config["serial"]["enabled"]:
        s = config["serial"]
        return f"Serial://{s['port']}@{s['baudrate']}"
    return "None"
