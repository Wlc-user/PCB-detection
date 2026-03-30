"""
根因分析API服务
基于DSSM的缺陷根因定位
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import base64
import json

# 导入根因分析模块
from defect_root_cause import RootCauseAnalyzer, DEFECT_PROCESS_MAPPING, ProcessStage


app = FastAPI(
    title="PCB缺陷根因定位系统",
    description="YOLOv8缺陷检测 + DSSM双塔匹配根因分析",
    version="2.0.0"
)

# 全局分析器
analyzer = None


@app.on_event("startup")
async def startup():
    global analyzer
    analyzer = RootCauseAnalyzer()


class RootCauseResponse(BaseModel):
    image: str
    total_defects: int
    detections: List[dict]
    process_alerts: List[dict]
    recommendation: str


@app.get("/")
async def root():
    return {
        "name": "PCB缺陷根因定位系统",
        "version": "2.0.0",
        "description": "YOLOv8 + DSSM双塔匹配",
        "endpoints": {
            "根因分析": "/analyze (POST)",
            "工序列表": "/processes (GET)",
            "缺陷映射": "/mapping (GET)"
        }
    }


@app.get("/processes")
async def get_processes():
    """获取所有工序列表"""
    return {
        "processes": [stage.value for stage in ProcessStage]
    }


@app.get("/mapping")
async def get_mapping():
    """获取缺陷-工序映射关系"""
    mapping = {}
    for defect, info in DEFECT_PROCESS_MAPPING.items():
        mapping[defect] = {
            "primary": [p.value for p in info["primary"]],
            "secondary": [p.value for p in info["secondary"]],
            "symptoms": info["symptoms"]
        }
    return mapping


@app.post("/analyze", response_model=RootCauseResponse)
async def analyze_defect(
    file: UploadFile = File(...),
    conf_threshold: Optional[float] = 0.15
):
    """
    缺陷检测 + 根因分析
    
    返回:
    - 缺陷检测结果
    - 每个缺陷的可能工序
    - 工序异常预警
    - 处理建议
    """
    if analyzer is None:
        raise HTTPException(status_code=503, detail="分析器未初始化")
    
    # 读取图片
    contents = await file.read()
    
    # 保存临时文件
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    
    try:
        # 分析
        result = analyzer.detect_and_analyze(tmp_path, conf_threshold)
        return result
    finally:
        os.unlink(tmp_path)


@app.post("/analyze/base64")
async def analyze_base64(image_data: str, conf_threshold: Optional[float] = 0.15):
    """Base64图片分析"""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="分析器未初始化")
    
    import tempfile
    import os
    import cv2
    import numpy as np
    
    # 解码Base64
    img_bytes = base64.b64decode(image_data)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # 保存临时文件
    tmp_path = tempfile.mktemp(suffix='.jpg')
    cv2.imwrite(tmp_path, img)
    
    try:
        result = analyzer.detect_and_analyze(tmp_path, conf_threshold)
        return result
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
