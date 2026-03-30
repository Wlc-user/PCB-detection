# 部署指南

## 本地部署

### 方式1：直接运行
```bash
# 安装依赖
pip install -r requirements.txt

# 启动API服务
python defect_detection_api.py

# 访问 http://localhost:8000/docs
```

### 方式2：Docker部署
```bash
# 构建镜像
docker build -t pcb-detector .

# 运行容器
docker run -p 8000:8000 pcb-detector

# 或使用docker-compose
docker-compose up -d
```

---

## 云端部署

### 阿里云/腾讯云
```bash
# 1. 构建镜像
docker build -t pcb-detector .

# 2. 推送镜像到镜像仓库
docker tag pcb-detector registry.cn-shenzhen.aliyuncs.com/myproject/pcb-detector:latest
docker push registry.cn-shenzhen.aliyuncs.com/myproject/pcb-detector:latest

# 3. 在云服务器创建实例并运行
docker run -d -p 8000:8000 registry.cn-shenzhen.aliyuncs.com/myproject/pcb-detector:latest
```

---

## API调用示例

```python
import requests

# 上传图片检测
with open('test.jpg', 'rb') as f:
    resp = requests.post(
        'http://localhost:8000/detect',
        files={'file': f},
        data={'conf_threshold': 0.15}
    )
    print(resp.json())
```

---

## 性能优化

| 优化项 | 方法 |
|--------|------|
| GPU加速 | 使用CUDA镜像 |
| 模型量化 | 导出TensorRT |
| 批量推理 | 修改batch_size |
| 缓存 | 添加Redis缓存 |
