# 万张数据集标注平台

## 功能特性

- **多种标注类型**: 2D边界框、多边形分割、关键点
- **AI辅助标注**: 支持SAM自动分割、YOLO预检测
- **团队协作**: 用户管理、任务分配、审核流程
- **批量导入导出**: 支持YOLO、COCO格式
- **万张级支持**: 数据库优化、分页加载

## 快速启动

### 1. 安装依赖

```bash
cd e:/pyspace/opencv
pip install flask flask-cors opencv-python numpy ultralytics
```

### 2. 启动服务

```bash
# 基本启动
python annotation_platform.py

# 初始化示例数据
python annotation_platform.py --init

# 指定端口和目录
python annotation_platform.py --port 8080 --dir ./my_data
```

### 3. 访问平台

打开浏览器: http://localhost:5000

- 用户名: `admin`
- 密码: `admin123`

## 使用流程

### 创建项目
1. 点击侧边栏「+ 新建项目」
2. 输入项目名称和描述
3. 选择标注类型（框/多边形/关键点）

### 导入图像
1. 选择任务，点击「导入」
2. 输入图像目录路径
3. 支持 jpg/png/bmp/tiff 格式

### 开始标注
1. 点击图像进入标注界面
2. 选择类别（快捷键1-9）
3. 用鼠标拖动绘制边界框
4. 按 `Ctrl+S` 保存

### 导出数据
1. 点击「导出」
2. 选择格式（YOLO/COCO）
3. 获取标注文件和图像

## 快捷键

| 快捷键 | 功能 |
|--------|------|
| `1-9` | 切换类别 |
| `Del` | 删除标注 |
| `Ctrl+S` | 保存 |
| `← →` | 切换图像 |
| `Space` | 下一张 |

## AI辅助功能

### SAM自动分割
```bash
# 下载SAM模型
# https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# 放置到当前目录或指定路径
```

### YOLO预检测
```bash
# 使用YOLOv8自动检测物体
# 减少70%标注工作量
```

## 数据格式

### YOLO格式
```
output/
├── labels/
│   ├── image1.txt  # 每张图一个txt
│   └── image2.txt
└── images/
    ├── image1.jpg
    └── image2.jpg
```

### COCO格式
```json
{
  "images": [...],
  "annotations": [...],
  "categories": [...]
}
```

## 目录结构

```
annotation_data/
├── platform.db        # SQLite数据库
├── datasets/          # 原始图像
├── thumbnails/        # 缩略图
└── exports/          # 导出数据
```

## API接口

```
GET  /api/stats                    # 平台统计
GET  /api/projects                 # 项目列表
POST /api/projects                 # 创建项目
GET  /api/tasks?project_id=1      # 任务列表
POST /api/tasks                   # 创建任务
POST /api/tasks/1/import          # 导入图像
GET  /api/tasks/1/images          # 图像列表
GET  /api/images/1                # 图像详情
POST /api/images/1/annotations    # 添加标注
POST /api/tasks/1/export          # 导出数据
```

## 推荐配置

### 小规模 (< 1000张)
- 单机部署
- 使用labelme或LabelImg

### 中等规模 (1000 - 10000张)
- 本平台
- 开启AI辅助
- 2-3人协作

### 大规模 (> 10000张)
- 使用CVAT Enterprise
- 或 Scale AI / Labelbox
