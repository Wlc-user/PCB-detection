"""
截图工具 - 截取API文档和检测结果
"""
import os
import time
from playwright.sync_api import sync_playwright

def capture_api_docs():
    """截取API文档页面"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_viewport_size({"width": 1280, "height": 800})
        
        # 打开API文档
        page.goto("http://localhost:8000/docs")
        time.sleep(2)
        page.screenshot(path="portfolio/api_docs.png", full_page=True)
        print("已保存: portfolio/api_docs.png")
        
        browser.close()

def capture_api_json():
    """截取API JSON页面"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_viewport_size({"width": 1280, "height": 800})
        
        page.goto("http://localhost:8000/openapi.json")
        time.sleep(1)
        page.screenshot(path="portfolio/api_json.png", full_page=True)
        print("已保存: portfolio/api_json.png")
        
        browser.close()

def capture_root():
    """截取根路径页面"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_viewport_size({"width": 1280, "height": 800})
        
        page.goto("http://localhost:8000/")
        time.sleep(1)
        page.screenshot(path="portfolio/api_root.png", full_page=True)
        print("已保存: portfolio/api_root.png")
        
        browser.close()

def capture_test_image():
    """测试单图检测"""
    import requests
    import json
    
    url = "http://localhost:8000/detect"
    
    with open("yolo_pcb_dataset/images/test/01_missing_hole_02.jpg", "rb") as f:
        files = {"file": f}
        resp = requests.post(url, files=files)
        result = resp.json()
    
    print("检测结果:", json.dumps(result, indent=2, ensure_ascii=False))
    return result

if __name__ == "__main__":
    # 创建portfolio目录
    os.makedirs("portfolio", exist_ok=True)
    
    # 截图
    print("开始截图...")
    capture_root()
    capture_api_docs()
    capture_api_json()
    
    print("\n测试检测API...")
    result = capture_test_image()
    
    print("\n完成!")
