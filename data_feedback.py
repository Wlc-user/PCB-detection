"""
数据回流系统
工厂部署后自动收集数据，用于持续优化
"""
import cv2
import sqlite3
import json
import time
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import shutil
import numpy as np


class DataCollector:
    """
    数据采集器 - 自动收集检测数据
    
    收集内容:
    - 原始图片
    - 检测结果 (OK/NG)
    - 缺陷信息
    - 时间/板号
    - 人工确认结果 (用于反馈学习)
    """
    
    def __init__(self, data_dir: str = "factory_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 子目录
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        self.images_dir.mkdir(exist_ok=True)
        self.labels_dir.mkdir(exist_ok=True)
        
        # 数据库
        self.db_path = self.data_dir / "inspection_data.db"
        self.init_db()
        
        # 配置
        self.image_counter = 0
        
        print(f"数据采集器初始化完成")
        print(f"  数据目录: {self.data_dir}")
        print(f"  数据库: {self.db_path}")
    
    def init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 检测记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inspection_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                board_id TEXT,
                timestamp TEXT,
                result TEXT,
                defect_count INTEGER,
                defects TEXT,
                image_path TEXT,
                human_confirmed INTEGER DEFAULT -1,
                used_for_training INTEGER DEFAULT 0,
                notes TEXT
            )
        """)
        
        # 统计表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                total_count INTEGER DEFAULT 0,
                ok_count INTEGER DEFAULT 0,
                ng_count INTEGER DEFAULT 0,
                confirmed_defect INTEGER DEFAULT 0,
                confirmed_ok INTEGER DEFAULT 0,
                exported_count INTEGER DEFAULT 0
            )
        """)
        
        conn.commit()
        conn.close()
    
    def collect(self, 
                image, 
                board_id: str, 
                result: str,
                defects: List[Dict],
                save_image: bool = True) -> str:
        """
        收集一条检测数据
        
        Args:
            image: numpy图片
            board_id: 板号
            result: OK/NG/ERROR
            defects: 缺陷列表
            save_image: 是否保存图片
            
        Returns:
            image_path: 保存的图片路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 生成文件名
        date_str = datetime.now().strftime("%Y%m%d")
        self.image_counter += 1
        filename = f"{date_str}_{self.image_counter:05d}_{board_id}_{result}.jpg"
        
        image_path = None
        if save_image:
            image_path = self.images_dir / filename
            cv2.imwrite(str(image_path), image)
        
        # 保存到数据库
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO inspection_records 
            (board_id, timestamp, result, defect_count, defects, image_path)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            board_id,
            timestamp,
            result,
            len(defects),
            json.dumps(defects, ensure_ascii=False),
            str(image_path) if image_path else None
        ))
        
        record_id = cursor.lastrowid
        
        # 更新每日统计
        self.update_daily_stats(date_str, result)
        
        conn.commit()
        conn.close()
        
        return str(image_path) if image_path else ""
    
    def update_daily_stats(self, date_str: str, result: str):
        """更新每日统计"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR IGNORE INTO daily_stats (date) VALUES (?)
        """, (date_str,))
        
        if result == "OK":
            cursor.execute("""
                UPDATE daily_stats 
                SET total_count = total_count + 1,
                    ok_count = ok_count + 1
                WHERE date = ?
            """, (date_str,))
        elif result == "NG":
            cursor.execute("""
                UPDATE daily_stats 
                SET total_count = total_count + 1,
                    ng_count = ng_count + 1
                WHERE date = ?
            """, (date_str,))
        
        conn.commit()
        conn.close()
    
    def confirm_result(self, record_id: int, is_defect: bool, notes: str = ""):
        """
        人工确认结果 (用于反馈学习)
        
        Args:
            record_id: 记录ID
            is_defect: 是否真的是缺陷
            notes: 备注
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        confirmed = 1 if is_defect else 0
        
        cursor.execute("""
            UPDATE inspection_records 
            SET human_confirmed = ?, notes = ?
            WHERE id = ?
        """, (confirmed, notes, record_id))
        
        # 更新统计
        date_str = datetime.now().strftime("%Y%m%d")
        if is_defect:
            cursor.execute("""
                UPDATE daily_stats SET confirmed_defect = confirmed_defect + 1
                WHERE date = ?
            """, (date_str,))
        else:
            cursor.execute("""
                UPDATE daily_stats SET confirmed_ok = confirmed_ok + 1
                WHERE date = ?
            """, (date_str,))
        
        conn.commit()
        conn.close()
    
    def get_stats(self, days: int = 7) -> Dict:
        """获取最近N天统计"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM daily_stats 
            ORDER BY date DESC 
            LIMIT ?
        """, (days,))
        
        rows = cursor.fetchall()
        conn.close()
        
        stats = {
            "total": 0,
            "ok": 0,
            "ng": 0,
            "confirmed_defect": 0,
            "confirmed_ok": 0
        }
        
        for row in rows:
            stats["total"] += row[1]
            stats["ok"] += row[2]
            stats["ng"] += row[3]
            stats["confirmed_defect"] += row[4]
            stats["confirmed_ok"] += row[5]
        
        return stats
    
    def export_for_training(self, 
                           min_samples: int = 50,
                           include_confirmed_ng: bool = True,
                           include_confirmed_ok: bool = False) -> Dict:
        """
        导出数据用于模型训练
        
        Args:
            min_samples: 每类最少样本数
            include_confirmed_ng: 包含确认的NG样本
            include_confirmed_ok: 包含确认的OK样本
            
        Returns:
            导出统计
        """
        export_dir = self.data_dir / "export" / datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir.mkdir(parents=True, exist_ok=True)
        
        export_images = export_dir / "images"
        export_labels = export_dir / "labels"
        export_images.mkdir(exist_ok=True)
        export_labels.mkdir(exist_ok=True)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 查询需要导出的记录
        query = """
            SELECT id, image_path, result, defects, human_confirmed 
            FROM inspection_records 
            WHERE image_path IS NOT NULL
        """
        
        conditions = []
        if include_confirmed_ng:
            conditions.append("human_confirmed = 1")
        if include_confirmed_ok:
            conditions.append("human_confirmed = 0")
        
        if conditions:
            query += " AND (" + " OR ".join(conditions) + ")"
        
        cursor.execute(query)
        records = cursor.fetchall()
        
        # 按缺陷类型分类
        defect_samples = {}
        ok_samples = []
        
        for record in records:
            record_id, image_path, result, defects_json, confirmed = record
            
            if not image_path or not os.path.exists(image_path):
                continue
            
            # 解析缺陷
            try:
                defects = json.loads(defects_json) if defects_json else []
            except:
                defects = []
            
            if confirmed == 1:  # 确认是缺陷
                # 提取缺陷类型
                for defect in defects:
                    cls = defect.get("class", "unknown")
                    if cls not in defect_samples:
                        defect_samples[cls] = []
                    defect_samples[cls].append((record_id, image_path))
            
            elif confirmed == 0 and include_confirmed_ok:  # 确认是OK
                ok_samples.append((record_id, image_path))
        
        # 复制文件
        exported_count = 0
        for cls, samples in defect_samples.items():
            # 每类取最多min_samples个
            samples = samples[:min_samples]
            
            cls_dir = export_labels / cls
            cls_dir.mkdir(exist_ok=True)
            
            for record_id, src_path in samples:
                # 复制图片
                dst_img = export_images / f"{cls}_{exported_count:05d}.jpg"
                shutil.copy(src_path, dst_img)
                
                # 复制标签 (如果有)
                label_file = Path(src_path).with_suffix('.txt')
                if label_file.exists():
                    dst_label = cls_dir / f"{cls}_{exported_count:05d}.txt"
                    shutil.copy(label_file, dst_label)
                
                # 标记已导出
                cursor.execute("""
                    UPDATE inspection_records SET used_for_training = 1
                    WHERE id = ?
                """, (record_id,))
                
                exported_count += 1
        
        # 处理OK样本
        if ok_samples:
            ok_samples = ok_samples[:min_samples]
            ok_dir = export_labels / "OK"
            ok_dir.mkdir(exist_ok=True)
            
            for record_id, src_path in ok_samples:
                dst_img = export_images / f"OK_{exported_count:05d}.jpg"
                shutil.copy(src_path, dst_img)
                
                cursor.execute("""
                    UPDATE inspection_records SET used_for_training = 1
                    WHERE id = ?
                """, (record_id,))
                
                exported_count += 1
        
        conn.commit()
        conn.close()
        
        # 生成导出报告
        report = {
            "export_dir": str(export_dir),
            "total_samples": exported_count,
            "defect_types": {cls: len(samples[:min_samples]) 
                           for cls, samples in defect_samples.items()},
            "ok_samples": len(ok_samples) if include_confirmed_ok else 0
        }
        
        print(f"数据导出完成: {exported_count} 样本")
        print(f"导出目录: {export_dir}")
        
        return report


class Data回流系统:
    """集成数据采集 + 导出 + 可视化"""
    
    def __init__(self, data_dir: str = "factory_data"):
        self.collector = DataCollector(data_dir)
    
    def 自动采集(self, image, board_id: str, result: str, defects: List[Dict]):
        """自动采集检测数据"""
        return self.collector.collect(image, board_id, result, defects)
    
    def 人工确认(self, record_id: int, is_defect: bool, notes: str = ""):
        """人工确认结果"""
        self.collector.confirm_result(record_id, is_defect, notes)
    
    def 查看统计(self, days: int = 7) -> Dict:
        """查看最近N天统计"""
        return self.collector.get_stats(days)
    
    def 导出训练数据(self, min_samples: int = 50) -> Dict:
        """导出用于训练"""
        return self.collector.export_for_training(min_samples)
    
    def 生成报告(self) -> str:
        """生成数据报告"""
        stats = self.collector.get_stats(30)
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    数据回流统计报告 (最近30天)                ║
╠══════════════════════════════════════════════════════════════╣
║  总检测数量:    {stats['total']:>8}                                  ║
║  OK数量:       {stats['ok']:>8}                                  ║
║  NG数量:       {stats['ng']:>8}                                  ║
║  确认缺陷:     {stats['confirmed_defect']:>8}                                  ║
║  确认正常:     {stats['confirmed_ok']:>8}                                  ║
╠══════════════════════════════════════════════════════════════╣
║  预估存储:     {stats['total'] * 0.5:>6.1f} MB (按0.5MB/张)                   ║
╚══════════════════════════════════════════════════════════════╝
        """
        return report


# 使用示例
if __name__ == "__main__":
    # 创建回流系统
    回流 = Data回流系统()
    
    print("="*60)
    print("数据回流系统 - 测试")
    print("="*60)
    
    # 1. 模拟采集数据
    print("\n[1] 模拟采集数据...")
    import glob
    test_images = glob.glob("yolo_pcb_dataset/images/test/*.jpg")[:5]
    
    for i, img_path in enumerate(test_images):
        img = cv2.imread(img_path)
        board_id = f"PCB_{i+1:04d}"
        result = "NG" if i < 3 else "OK"
        defects = [{"class": "missing_hole", "confidence": 0.8}] if result == "NG" else []
        
        回流.自动采集(img, board_id, result, defects)
    
    # 2. 查看统计
    print("\n[2] 统计信息:")
    stats = 回流.查看统计()
    print(f"  总数: {stats['total']}")
    print(f"  OK: {stats['ok']}")
    print(f"  NG: {stats['ng']}")
    
    # 3. 模拟人工确认 (假设record_id=1是缺陷)
    print("\n[3] 模拟人工确认...")
    回流.人工确认(1, is_defect=True, notes="确实有缺失孔洞")
    回流.人工确认(2, is_defect=False, notes="误检，实际正常")
    
    # 4. 导出训练数据
    print("\n[4] 导出训练数据:")
    report = 回流.导出训练数据(min_samples=10)
    print(f"  导出数量: {report['total_samples']}")
    
    # 5. 生成报告
    print(回流.生成报告())
