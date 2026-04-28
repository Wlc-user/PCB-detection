"""
Commercial Platform - API Test Script
"""
import requests
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://localhost:5000/api/v1"

def test_apis():
    print("=" * 60)
    print("Commercial Annotation Platform - API Test")
    print("=" * 60)
    
    # 1. Health Check
    print("\n[1] Health Check...")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"    Status: {r.status_code}")
        print(f"    Response: {r.json()}")
    except Exception as e:
        print(f"    [FAIL] Service not started: {e}")
        print("\nRun: python commercial_platform.py --init")
        return
    
    # 2. View Plans
    print("\n[2] Pricing Plans...")
    r = requests.get(f"{BASE_URL}/plans")
    plans = r.json()
    for plan, info in plans.items():
        print(f"    {info['name']}: ${info['price']}/mo, {info['images_per_month']} images/mo")
    
    # 3. Register New Tenant
    print("\n[3] Register Tenant...")
    r = requests.post(f"{BASE_URL}/auth/register", json={
        "tenant_name": "TestCompany",
        "username": "testuser",
        "email": "test@example.com",
        "password": "test123456"
    })
    print(f"    Status: {r.status_code}")
    print(f"    Response: {r.json()}")
    
    # 4. Login
    print("\n[4] User Login...")
    r = requests.post(f"{BASE_URL}/auth/login", json={
        "username": "admin",
        "password": "admin123"
    })
    if r.status_code == 200:
        data = r.json()
        token = data.get("token", "")
        print(f"    [OK] Login success!")
        print(f"    Token: {token[:40]}...")
        
        headers = {"Authorization": f"Bearer {token}"}
        
        # 5. View Usage
        print("\n[5] Quota Usage...")
        r = requests.get(f"{BASE_URL}/usage", headers=headers)
        usage = r.json()
        print(f"    Plan: {usage.get('plan_name')}")
        print(f"    Images: {usage.get('images_this_month')}/{usage.get('images_limit')}")
        print(f"    Storage: {usage.get('storage_used_gb')}/{usage.get('storage_limit_gb')} GB")
        
        # 6. Create Project
        print("\n[6] Create Project...")
        r = requests.post(f"{BASE_URL}/projects", headers=headers, json={
            "name": "PCB Detection Dataset",
            "description": "For training PCB defect detection model"
        })
        print(f"    Status: {r.status_code}")
        print(f"    Response: {r.json()}")
        
        # 7. List Projects
        print("\n[7] List Projects...")
        r = requests.get(f"{BASE_URL}/projects", headers=headers)
        print(f"    Total: {len(r.json())} projects")
        for p in r.json():
            print(f"    - {p['name']} (id={p['id']})")
        
    else:
        print(f"    [FAIL] Login failed: {r.text}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_apis()
