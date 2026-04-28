# PortAI 工业通信系统

## 项目说明

实现上位机与下位机（PLC）的TCP通信，支持Modbus TCP协议。

## 快速启动

### 1. 启动 PLC 模拟器 (Python)
```bash
cd e:/pyspace/opencv/PortComm
python start_plc.py
```
- PLC监听端口: **5000**
- 协议: Modbus TCP
- 寄存器: 8个Holding Registers

### 2. 启动 上位机

**方案A: Python上位机 (推荐，立即可用)**
```bash
cd e:/pyspace/opencv/PortComm
python port_comm_gui.py
```

**方案B: C#上位机 (需要Visual Studio编译)**
```powershell
# 打开 Visual Studio
# 文件 -> 打开 -> 项目/解决方案
# 选择 PortComm.csproj
# Ctrl+F5 运行
```

## 系统架构

```
┌─────────────────┐      TCP:5000        ┌─────────────────┐
│   上位机         │ ←────────────────→  │  PLC模拟器       │
│                 │    Modbus TCP      │                 │
│  - 连接/断开     │ ←────────────────→  │  寄存器:         │
│  - 读取数据      │                     │  [0] 起重机状态   │
│  - 发送命令      │                     │  [1] 集装箱位置   │
│  - 监控界面      │                     │  [2] 卡车数量     │
└─────────────────┘                     │  [3] 船舶检测     │
                                        │  [4] 告警        │
                                        │  [5] 速度        │
                                        │  [6] 温度        │
                                        │  [7] 重量        │
                                        └─────────────────┘
```

## 寄存器说明

| 地址 | 名称 | 说明 | 示例值 |
|------|------|------|--------|
| 0 | CRANE_ST | 起重机状态 (0=停止, 1=运行, 2=故障) | 0-2 |
| 1 | CONT_POS | 集装箱位置 (0-100) | 0-100 |
| 2 | TRUCK_CNT | 卡车数量 | 0-5 |
| 3 | SHIP_DET | 船舶检测 (0=无, 1=有) | 0-1 |
| 4 | ALARM | 告警标志 (0=正常, 1=告警) | 0-1 |
| 5 | SPEED | 速度 (km/h) | 0-100 |
| 6 | TEMP | 温度 (°C) | 20-35 |
| 7 | WEIGHT | 重量 (kg) | 0-5000 |

## Modbus TCP 通信

### 读取寄存器 (Function 03)

**请求:**
```
00 01 00 00 00 06 01 03 00 00 00 08
```

**响应示例:**
```
00 01 00 00 00 11 01 03 10 00 00 00 0C 00 04 00 00 00 00 00 5B 00 1D 0C 72
```

## 文件清单

```
PortComm/
├── start_plc.py         PLC模拟器启动器
├── PLCSimulator.py       PLC模拟器核心
├── port_comm_gui.py      Python上位机界面
├── PortComm.cs          C#上位机源代码
├── PortComm.csproj      .NET项目文件
├── test_client.py       测试客户端
├── README.md            说明文档
└── build.bat            编译脚本
```

## C# 编译说明

如果您需要编译C#上位机:

1. **使用 Visual Studio**
   - 打开 Visual Studio
   - 文件 → 打开 → 项目/解决方案
   - 选择 `PortComm.csproj`
   - Ctrl+F5 运行

2. **使用命令行**
   ```powershell
   cd e:/pyspace/opencv/PortComm
   dotnet build -c Release
   dotnet run -c Release
   ```

## 依赖

- Python 3.8+
- .NET 9.0 SDK (仅C#版本需要)
