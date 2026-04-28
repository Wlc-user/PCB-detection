# -*- coding: utf-8 -*-
"""
Universal Data Annotation Platform - Full Industry Coverage
All-in-one tool for: Autonomous, Medical, Remote Sensing, Industrial, OCR, NLP, etc.
"""
import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from pathlib import Path

# ============================================================
# COMPLETE INDUSTRY CONFIGURATIONS
# ============================================================

DOMAINS = {
    
    # ===== 1. AUTONOMOUS DRIVING (Largest Market) =====
    'autonomous': {
        'name': '自动驾驶',
        'name_en': 'Autonomous Driving',
        'classes': [
            # 车辆类 (0-8)
            'car', 'truck', 'bus', 'van', 'motorcycle', 'bicycle', 'tricycle', 'electric_cart', 'other_vehicle',
            # 行人 (9-12)
            'pedestrian', 'rider', 'person_sitting', 'person_lying',
            # 交通设施 (13-19)
            'traffic_light_red', 'traffic_light_yellow', 'traffic_light_green', 'traffic_sign', 
            'stop_line', 'crosswalk', 'speed_bump', 'parking_space',
            # 道路设施 (20-26)
            'pole', 'fence', 'wall', 'guardrail', 'bridge', 'tunnel', 'overpass',
            # 可行驶区域 (27-29)
            'drivable_area', 'lane_marking', 'curb',
        ],
        'color': [
            '#FF0000', '#CC0000', '#990000', '#FF3333', '#FF6666', '#FF9999', '#FFCCCC',
            '#00FF00', '#00CC00', '#99FF99',
            '#0000FF', '#0066FF', '#3399FF', '#66CCFF',
            '#FFFF00', '#FFCC00', '#FF9900', '#FF6600',
            '#00FFFF', '#00CCFF', '#0099FF', '#0066CC',
            '#FF00FF', '#CC00FF', '#9900FF', '#6600FF',
            '#000000', '#333333', '#666666', '#999999',
        ]
    },
    
    # ===== 2. MEDICAL IMAGING (High Demand) =====
    'medical': {
        'name': '医疗影像',
        'name_en': 'Medical Imaging',
        'classes': [
            # CT/X光 胸部 (0-9)
            'lung_nodule', 'lung_mass', 'pulmonary_embolism', 'pneumonia', 'tuberculosis',
            'mediastinal_mass', 'pleural_effusion', 'pneumothorax', 'cardiac_hypertrophy', 'aortic_aneurysm',
            # CT/X光 腹部 (10-19)
            'liver_tumor', 'liver_cyst', 'liver_metastasis', 'hepatic_steatosis', 'gallstone',
            'kidney_tumor', 'kidney_cyst', 'kidney_stone', 'adrenal_mass', 'splenic_mass',
            # CT/X光 其他 (20-29)
            'pancreatic_tumor', 'adrenal_tumor', 'retroperitoneal_mass', 'abdominal_aortic_aneurysm', 'bowel_obstruction',
            'bone_fracture', 'bone_lesion', 'bone_metastasis', 'spine_fracture', 'spine_deformity',
            # 超声 (30-39)
            'thyroid_nodule', 'thyroid_cancer', 'breast_mass', 'breast_cancer', 'breast_calcification',
            'ovarian_cyst', 'ovarian_tumor', 'uterine_fibroid', 'prostate_mass', 'testicular_mass',
            # 皮肤/眼底 (40-49)
            'skin_tumor', 'melanoma', 'basal_cell_carcinoma', 'squamous_cell_carcinoma', 'actinic_keratosis',
            'retinal_vessel', 'optic_disc', 'macula', 'retinal_detachment', 'diabetic_retinopathy',
            # 病理 (50-59)
            'benign_tumor', 'malignant_tumor', 'carcinoma_in_situ', 'glandular_cell', 'squamous_cell',
            'inflammatory_cell', 'necrotic_tissue', 'dysplastic_cell', 'metastatic_cell', 'polyp',
        ],
        'color': [
            '#FF6B6B', '#EE5A5A', '#DD4949', '#CC3838', '#BB2727', '#AA1616', '#990505', '#880000',
            '#4ECDC4', '#3EBDB4', '#2EADA4', '#1E9D94', '#0E8D84', '#007D74', '#006D64', '#005D54',
            '#45B7D1', '#35A7C1', '#2597B1', '#1587A1', '#057791', '#006781', '#005771', '#004761',
            '#96CEB4', '#86BE9A', '#76AE80', '#669E6A', '#568E5A', '#467E4A', '#366E3A', '#265E2A',
            '#FFEAA7', '#FFDA97', '#FFCA87', '#FFBA77', '#FFAA67', '#FF9A57', '#FF8A47', '#FF7A37',
            '#DDA0DD', '#CD90CD', '#BD80BD', '#AD70AD', '#9D609D', '#8D508D', '#7D407D', '#6D306D',
        ]
    },
    
    # ===== 3. REMOTE SENSING =====
    'remote_sensing': {
        'name': '遥感影像',
        'name_en': 'Remote Sensing',
        'classes': [
            # 建筑 (0-9)
            'building', 'commercial_building', 'residential_building', 'industrial_building', 'warehouse',
            'hospital', 'school', 'stadium', 'tower', 'construction',
            # 交通 (10-15)
            'road', 'highway', 'railway', 'bridge_rs', 'airport', 'port',
            # 自然 (16-25)
            'forest', 'grassland', 'cropland', 'orchard', 'vegetation',
            'water_body', 'river', 'lake', 'sea', 'pond',
            'bare_land', 'desert', 'beach', 'wetland', 'snow',
            # 变化检测 (26-29)
            'new_building', 'demolished', 'road_change', 'land_use_change',
        ],
        'color': [
            '#FF4444', '#DD3333', '#CC2222', '#BB1111', '#AA0000', '#990000', '#880000', '#770000', '#660000', '#550000',
            '#444444', '#333333', '#222222', '#666666', '#777777', '#888888',
            '#00AA00', '#008800', '#006600', '#004400', '#00CC00', '#00BB00', '#009900', '#00DD00', '#00EE00',
            '#0000FF', '#0000CC', '#000099', '#000066', '#0000AA', '#0000BB', '#0000DD', '#0000EE',
            '#AAAAAA', '#BBBBBB', '#CCCCCC', '#DDDDDD', '#EEEEEE',
        ]
    },
    
    # ===== 4. INDUSTRIAL DEFECT (PCB, Welding, etc.) =====
    'industrial': {
        'name': '工业缺陷',
        'name_en': 'Industrial Defect',
        'classes': [
            # PCB缺陷 (0-15)
            'pcb_scratch', 'pcb_bridge', 'pcb_missing_hole', 'pcb_excess_solder', 'pcb_insufficient_solder',
            'pcb_shift', 'pcb_tombstone', 'pcb_foreign', 'pcb_stain', 'pcb_oxidization',
            'pcb_copper_exposure', 'pcb_delamination', 'pcb_void', 'pcb_crack', 'pcb_sid_band', 'pcb_alignment',
            # 焊接缺陷 (16-25)
            'weld_porosity', 'weld_slag', 'weld_crack', 'weld_incomplete_fusion', 'weld_undercut',
            'weld_overlap', 'weld_spatter', 'weld_burn_through', 'weld_lack_of_penetration', 'weld_excessive_penetration',
            # 表面缺陷 (26-39)
            'surface_scratch', 'surface_dent', 'surface_crack', 'surface_pit', 'surface_burr',
            'surface_contamination', 'surface_discoloration', 'surface_corrosion', 'surface_painting_defect', 'surface_coating_defect',
            'surface_fingerprint', 'surface_dust', 'surface_oil', 'surface_water_spot',
            # 金属缺陷 (40-49)
            'metal_crack', 'metal_fatigue', 'metal_corrosion', 'metal_wear', 'metal_deformation',
            'metal_weld_defect', 'metal_inclusion', 'metal_laminations', 'metal_seams', 'metal_folds',
            # 织物/皮革 (50-59)
            'fabric_hole', 'fabric_thread', 'fabric_stain', 'fabric_color_defect', 'fabric_weave_defect',
            'leather_scratch', 'leather_crack', 'leather_color_difference', 'leather_crease', 'leather_puncture',
        ],
        'color': [
            '#FF0000', '#FF1111', '#FF2222', '#FF3333', '#FF4444', '#FF5555', '#FF6666', '#FF7777',
            '#FF8888', '#FF9999', '#FFAAAA', '#FFBBBB', '#FFCCCC', '#FFDDDD', '#FFEEEE', '#FFFFFF',
            '#00FF00', '#00EE00', '#00DD00', '#00CC00', '#00BB00', '#00AA00', '#009900', '#008800',
            '#007700', '#006600', '#005500', '#004400', '#003300', '#002200', '#001100', '#000000',
            '#0000FF', '#0011FF', '#0022FF', '#0033FF', '#0044FF', '#0055FF', '#0066FF', '#0077FF',
            '#0088FF', '#0099FF', '#00AAFF', '#00BBFF', '#00CCFF', '#00DDFF', '#00EEFF', '#00FFFF',
            '#FFFF00', '#FFEE00', '#FFDD00', '#FFCC00', '#FFBB00', '#FFAA00', '#FF9900', '#FF8800',
        ]
    },
    
    # ===== 5. OCR TEXT RECOGNITION =====
    'ocr': {
        'name': 'OCR文字',
        'name_en': 'OCR Text Recognition',
        'classes': [
            # 文档 (0-9)
            'document_title', 'document_body', 'document_caption', 'document_footnote', 'document_header',
            'document_page_number', 'document_heading', 'document_paragraph', 'document_list', 'document_table',
            # 手写 (10-19)
            'handwritten_text', 'handwritten_signature', 'handwritten_number', 'handwritten_form', 'handwritten_note',
            'handwritten_address', 'handwritten_name', 'handwritten_date', 'handwritten_amount', 'handwritten_check',
            # 印刷 (20-29)
            'printed_text', 'printed_title', 'printed_caption', 'printed_sidebar', 'printed_advertisement',
            'receipt_text', 'invoice_text', 'contract_text', 'newspaper_text', 'book_text',
            # 特殊 (30-39)
            'chinese_text', 'english_text', 'number_text', 'mixed_text', 'special_character',
            'logo', 'brand_name', 'product_name', 'barcode', 'qr_code',
            # 车牌/卡证 (40-49)
            'license_plate', 'id_card_text', 'bank_card_text', 'passport_text', 'business_card_text',
            'vehicle_registration', 'driver_license', 'medical_record', 'prescription', 'medical_report',
        ],
        'color': [
            '#E74C3C', '#C0392B', '#A93226', '#922B21', '#7B241C', '#641E16', '#922B21', '#7B241C',
            '#E67E22', '#D35400', '#CA6F1E', '#BA4A00', '#A04000', '#873600', '#6E2C00', '#5D2E00',
            '#3498DB', '#2980B9', '#2471A3', '#1F618D', '#1A5276', '#154360', '#0E4D64', '#0A3D62',
            '#2ECC71', '#27AE60', '#229954', '#1E8449', '#196F3D', '#145A32', '#117A65', '#0E6655',
            '#9B59B6', '#8E44AD', '#7D3C98', '#6C3483', '#5B2C6F', '#4A235A', '#4A235A', '#3D2963',
            '#F39C12', '#D68910', '#B7950B', '#9A7D0A', '#7D6608', '#6E2C00', '#935116', '#7E5109',
            '#1ABC9C', '#16A085', '#138D75', '#117A65', '#0E6655', '#0B5345', '#082345', '#073D3F',
        ]
    },
    
    # ===== 6. HUMAN POSE ESTIMATION =====
    'pose': {
        'name': '人体姿态',
        'name_en': 'Human Pose Estimation',
        'keypoints': [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
            # 额外关键点 (17-25)
            'neck', 'head_top', 'chin', 'left_heel', 'right_heel',
            'left_foot_tip', 'right_foot_tip', 'left_hand', 'right_hand',
        ],
        'skeleton': [
            (0,1), (0,2), (1,3), (2,4),  # 头
            (5,6), (5,7), (7,9), (6,8), (8,10),  # 上身
            (5,11), (6,12), (11,12),  # 躯干
            (11,13), (13,15), (12,14), (14,16),  # 下身
            (17,18), (18,0),  # 脖子到头
            (15,26), (16,27),  # 脚
            (9,28), (10,29),  # 手
        ],
        'color': ['#FF0000', '#FF1100', '#FF2200', '#FF3300', '#FF4400', '#FF5500', '#FF6600', '#FF7700', '#FF8800', '#FF9900',
                  '#FFAA00', '#FFBB00', '#FFCC00', '#FFDD00', '#FFEE00', '#FFFF00', '#EEFF00', '#DDFF00', '#CCFF00', '#BBFF00',
                  '#AAFF00', '#99FF00', '#88FF00', '#77FF00', '#66FF00', '#55FF00', '#44FF00', '#33FF00', '#22FF00', '#11FF00']
    },
    
    # ===== 7. FACE RECOGNITION =====
    'face': {
        'name': '人脸识别',
        'name_en': 'Face Recognition',
        'classes': [
            # 人脸 (0-5)
            'face_front', 'face_profile', 'face_half', 'occluded_face', 'blur_face', 'small_face',
            # 五官 (6-17)
            'left_eye', 'right_eye', 'left_eyebrow', 'right_eyebrow', 'nose', 'mouth', 'upper_lip', 'lower_lip',
            'left_ear', 'right_ear', 'left_cheek', 'right_cheek', 'chin', 'forehead',
            # 属性 (18-27)
            'face_with_glasses', 'face_with_sunglasses', 'face_with_mask', 'face_with_hat', 'face_with_beard',
            'face_with_mustache', 'face_with_hair', 'face_emotion_happy', 'face_emotion_sad', 'face_emotion_angry',
            # 特征点 (28-95)
            'landmark_5', 'landmark_19', 'landmark_29', 'landmark_68', 'landmark_98', 'landmark_106',
        ],
        'color': [
            '#FF6B6B', '#FF8787', '#FFA0A0', '#FFB5B5', '#FFCACA', '#FFDFDF',
            '#4ECDC4', '#5ED9D0', '#6EE5DC', '#7EF1E8', '#8EFDF4', '#9EF9FF',
            '#45B7D1', '#55C3DB', '#65CFE5', '#75DBEF', '#85E7F9', '#95F3FF',
            '#96CEB4', '#A6D4C0', '#B6DACC', '#C6E0D8', '#D6E6E4', '#E6ECF0',
            '#FFEAA7', '#FFEDB3', '#FFF0BF', '#FFF3CB', '#FFF6D7', '#FFF9E3',
            '#DDA0DD', '#E3ADE3', '#E9BAE9', '#EFC7EF', '#F5D4F5', '#FBE1FB',
        ]
    },
    
    # ===== 8. GENERAL OBJECTS =====
    'general': {
        'name': '通用物体',
        'name_en': 'General Objects',
        'classes': [
            # 人 (0-9)
            'person', 'man', 'woman', 'child', 'baby',
            'crowd', 'group', 'team', 'couple', 'family',
            # 动物 (10-29)
            'dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'pig', 'elephant', 'lion', 'tiger',
            'bear', 'monkey', 'rabbit', 'snake', 'fish', 'chicken', 'duck', 'bird_wild', 'insect', 'butterfly',
            # 食物 (30-44)
            'fruit', 'apple', 'banana', 'orange', 'grape', 'strawberry', 'watermelon', 'pineapple',
            'vegetable', 'carrot', 'tomato', 'potato', 'onion', 'broccoli', 'lettuce',
            'food_dish', 'pizza', 'burger', 'rice', 'bread', 'cake', 'cookie', 'coffee', 'wine', 'juice',
            # 物品 (45-79)
            'phone', 'laptop', 'computer', 'keyboard', 'mouse', 'monitor', 'printer', 'camera', 'tv', 'remote',
            'book', 'magazine', 'newspaper', 'paper', 'cardboard', 'plastic_bottle', 'glass', 'cup', 'plate', 'bowl',
            'chair', 'table', 'desk', 'sofa', 'bed', 'lamp', 'clock', 'plant', 'flower', 'tree',
            'car', 'bicycle', 'motorcycle', 'bus', 'train', 'boat', 'airplane', 'truck', 'suitcase', 'bag',
        ],
        'color': [
            '#FF4444', '#FF5555', '#FF6666', '#FF7777', '#FF8888', '#FF9999', '#FFAAAA', '#FFBBBB', '#FFCCCC', '#FFDDDD',
            '#44FF44', '#55FF55', '#66FF66', '#77FF77', '#88FF88', '#99FF99', '#AAAAFF', '#BBBBFF', '#CCCCFF', '#DDDDFF',
            '#FFFF44', '#FFFF55', '#FFFF66', '#FFFF77', '#FFFF88', '#FFFF99', '#FFFFAA', '#FFFFBB', '#FFFFCC', '#FFFFDD',
            '#FF44FF', '#FF55FF', '#FF66FF', '#FF77FF', '#FF88FF', '#FF99FF', '#FFAAFF', '#FFBBFF', '#FFCCFF', '#FFDDFF',
            '#44FFFF', '#55FFFF', '#66FFFF', '#77FFFF', '#88FFFF', '#99FFFF', '#AAFFFF', '#BBFFFF', '#CCFFFF', '#DDFFFF',
            '#FF8844', '#FF9955', '#FFAA66', '#FFBB77', '#FFCC88', '#FFDD99', '#FFEEAA', '#FFFFBB', '#FFFFCC', '#FFAACC',
        ]
    },
    
    # ===== 9. NLP TEXT ANNOTATION =====
    'nlp': {
        'name': 'NLP文本',
        'name_en': 'NLP Text Annotation',
        'classes': [
            # 通用实体 (0-19)
            'person', 'organization', 'location', 'country', 'city', 'province', 'district', 'address',
            'person_name', 'company_name', 'brand_name', 'product_name', 'food_name', 'disease_name', 'drug_name',
            # 时间/数量 (20-34)
            'date', 'time', 'duration', 'frequency', 'age', 'money', 'percentage', 'count', 'weight', 'height',
            'temperature', 'distance', 'area', 'volume', 'speed',
            # 事件/关系 (35-44)
            'event', 'accident', 'crime', 'natural_disaster', 'sports_event', 'political_event', 'cultural_event',
            'relation_org', 'relation_person', 'relation_family', 'relation_work',
            # 情感/观点 (45-49)
            'positive_sentiment', 'negative_sentiment', 'neutral_sentiment', 'opinion', 'aspect',
            # 自定义实体 (50-59)
            'custom_entity_1', 'custom_entity_2', 'custom_entity_3', 'custom_entity_4', 'custom_entity_5',
            'custom_entity_6', 'custom_entity_7', 'custom_entity_8', 'custom_entity_9', 'custom_entity_10',
        ],
        'color': [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#F8B500', '#E67E22', '#27AE60', '#2980B9', '#8E44AD', '#16A085', '#2ECC71', '#E74C3C', '#9B59B6', '#34495E',
            '#F39C12', '#D35400', '#1ABC9C', '#3498DB', '#E91E63', '#00BCD4', '#4CAF50', '#FF5722', '#673AB7', '#00BCD4',
            '#795548', '#9E9E9E', '#607D8B', '#FFEB3B', '#03A9F4', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#3F51B5',
            '#009688', '#8BC34A', '#FFC107', '#FF5722', '#607D8B', '#FF9800', '#00BCD4', '#8BC34A', '#795548', '#9E9E9E',
        ]
    },
    
    # ===== 10. VIDEO SURVEILLANCE =====
    'surveillance': {
        'name': '视频监控',
        'name_en': 'Video Surveillance',
        'classes': [
            # 安全相关 (0-14)
            'person_running', 'person_walking', 'person_standing', 'person_sitting', 'person_falling',
            'person_fighting', 'person_stealing', 'person_vandalism', 'person_loitering', 'person_abandoned_object',
            'vehicle_speeding', 'vehicle_stopped', 'vehicle_illegal_parking', 'vehicle_accident', 'vehicle_wrong_way',
            # 安检 (15-24)
            'baggage_left', 'baggage_removed', 'person_intrusion', 'perimeter_breach', 'gate_illegal_entry',
            'fire_detected', 'smoke_detected', 'flood_detected', 'crowd_gathering', 'queue_length',
            # 行为分析 (25-34)
            'abnormal_behavior', 'retrace_path', 'direction_violation', 'zone_intrusion', 'tailgating',
            'object_removed', 'object_left', 'scene_change', 'camera_obstruction', 'camera_defocused',
        ],
        'color': [
            '#FF0000', '#FF1100', '#FF2200', '#FF3300', '#FF4400', '#FF5500', '#FF6600', '#FF7700', '#FF8800', '#FF9900',
            '#FFAA00', '#FFBB00', '#FFCC00', '#FFDD00', '#FFEE00', '#FFFF00', '#EEFF00', '#DDFF00', '#CCFF00', '#BBFF00',
            '#00FF00', '#00EE00', '#00DD00', '#00CC00', '#00BB00', '#00AA00', '#009900', '#008800', '#007700', '#006600',
            '#0000FF', '#0011FF', '#0022FF', '#0033FF', '#0044FF', '#0055FF', '#0066FF', '#0077FF', '#0088FF', '#0099FF',
        ]
    },
    
    # ===== 11. SAR (Synthetic Aperture Radar) =====
    'sar': {
        'name': 'SAR影像',
        'name_en': 'SAR Imaging',
        'classes': [
            # 舰船 (0-9)
            'ship_large', 'ship_medium', 'ship_small', 'shipwreck', 'anchor_zone',
            # 建筑 (10-19)
            'building_sar', 'bridge_sar', 'road_sar', 'runway', 'harbor',
            # 地形 (20-29)
            'mountain', 'river_sar', 'coastline', 'ice', 'sea_ice',
            # 变化 (30-34)
            'new_construction', 'demolition', 'flood', 'earthquake_damage', 'deforestation',
        ],
        'color': [
            '#FF6B6B', '#FF8787', '#FFA0A0', '#FFB5B5', '#FFCACA', '#FFDADA', '#FFE8E8', '#FFF0F0', '#FFF5F5', '#FFFAFA',
            '#4ECDC4', '#5ED9D0', '#6EE5DC', '#7EF1E8', '#8EFDF4', '#9EF9FF', '#AEFDFF', '#BEFDFF', '#CEFDFF', '#DEFDFF',
            '#45B7D1', '#55C3DB', '#65CFE5', '#75DBEF', '#85E7F9', '#95F3FF', '#A5FAFF', '#B5FAFF', '#C5FAFF', '#D5FAFF',
            '#96CEB4', '#A6D4C0', '#B6DACC', '#C6E0D8', '#D6E6E4', '#E6ECF0', '#F0F5F0', '#F5FAF5', '#FAFFFA', '#FFFFFF',
        ]
    },
    
    # ===== 12.点云/3D =====
    'pointcloud': {
        'name': '点云3D',
        'name_en': 'Point Cloud / 3D',
        'classes': [
            # 3D车辆 (0-9)
            'car_3d', 'truck_3d', 'bus_3d', 'motorcycle_3d', 'bicycle_3d',
            'pedestrian_3d', 'rider_3d', 'other_vehicle_3d', 'unknown_3d', 'background_3d',
            # 室内物品 (10-19)
            'table_3d', 'chair_3d', 'sofa_3d', 'bed_3d', 'cabinet_3d',
            'desk_3d', 'shelf_3d', 'lamp_3d', 'plant_3d', 'appliance_3d',
            # 建筑 (20-29)
            'wall_3d', 'floor_3d', 'ceiling_3d', 'door_3d', 'window_3d',
            'stair_3d', 'beam_3d', 'column_3d', 'railing_3d', 'furniture_3d',
        ],
        'color': [
            '#FF0000', '#FF3300', '#FF6600', '#FF9900', '#FFCC00', '#FFFF00', '#CCFF00', '#99FF00', '#66FF00', '#33FF00',
            '#00FF00', '#00FF33', '#00FF66', '#00FF99', '#00FFCC', '#00FFFF', '#00CCFF', '#0099FF', '#0066FF', '#0033FF',
            '#0000FF', '#3300FF', '#6600FF', '#9900FF', '#CC00FF', '#FF00FF', '#FF00CC', '#FF0099', '#FF0066', '#FF0033',
        ]
    },
}


class UniversalAnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Universal Data Annotation Platform - Full Edition")
        self.root.geometry("1500x950")
        self.root.configure(bg='#1a1a2e')
        
        # State
        self.current_domain = 'autonomous'
        self.current_image = None
        self.current_image_path = None
        self.image_files = []
        self.current_index = 0
        
        self.annotations = []
        self.current_class = 0
        self.mode = 'bbox'
        
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_polygon = []
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        self.setup_ui()
        self.load_domain()
        
    def setup_ui(self):
        # Top toolbar
        toolbar = tk.Frame(self.root, bg='#16213e', height=60)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        tk.Label(toolbar, text="[ UNIVERSAL ANNOTATOR ]", font=('Consolas', 14, 'bold'),
                bg='#16213e', fg='#e94560').pack(side=tk.LEFT, padx=15, pady=10)
        
        # Domain selector
        tk.Label(toolbar, text="INDUSTRY:", bg='#16213e', fg='white').pack(side=tk.LEFT, padx=(10,5))
        
        self.domain_var = tk.StringVar()
        domain_names = [f"{d['name']} ({d['name_en']})" for d in DOMAINS.values()]
        self.domain_combo = ttk.Combobox(toolbar, textvariable=self.domain_var,
                                         values=domain_names, state='readonly', width=22, font=('Arial', 10))
        self.domain_combo.pack(side=tk.LEFT)
        self.domain_combo.bind('<<ComboboxSelected>>', self.on_domain_change)
        self.domain_combo.current(0)
        
        tk.Frame(toolbar, bg='#e94560', width=3).pack(side=tk.LEFT, fill=tk.Y, padx=15)
        
        # Buttons
        for text, cmd in [("Open Image", self.open_image), 
                          ("Open Folder", self.open_folder),
                          ("Save (Ctrl+S)", self.save_annotations)]:
            btn = tk.Button(toolbar, text=text, command=cmd, bg='#0f3460', fg='white',
                           font=('Arial', 9), relief=tk.FLAT, padx=10, pady=5)
            btn.pack(side=tk.LEFT, padx=3)
        
        tk.Frame(toolbar, bg='#e94560', width=3).pack(side=tk.LEFT, fill=tk.Y, padx=15)
        
        # Export
        tk.Label(toolbar, text="EXPORT:", bg='#16213e', fg='white').pack(side=tk.LEFT, padx=5)
        for fmt in ['YOLO', 'COCO', 'VOC']:
            btn = tk.Button(toolbar, text=fmt, command=lambda f=fmt: self.export_format(f),
                           bg='#16213e', fg='#e94560', font=('Arial', 9), relief=tk.FLAT, padx=10, pady=5)
            btn.pack(side=tk.LEFT, padx=2)
        
        # Main area
        main_frame = tk.Frame(self.root, bg='#1a1a2e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left canvas
        canvas_frame = tk.Frame(main_frame, bg='#1a1a2e')
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='#2d2d44', width=1050, height=850, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        
        # Right panel
        right_panel = tk.Frame(main_frame, bg='#16213e', width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_panel.pack_propagate(False)
        
        # Mode selection
        mode_frame = tk.LabelFrame(right_panel, text="[ Mode ]", bg='#16213e', fg='#e94560',
                                   font=('Arial', 10, 'bold'))
        mode_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.mode_var = tk.StringVar(value='bbox')
        for mode, text in [('bbox', 'BBox [B]'), ('polygon', 'Polygon [P]'), ('keypoint', 'Keypoint [K]')]:
            tk.Radiobutton(mode_frame, text=text, variable=self.mode_var, value=mode,
                          command=lambda m=mode: self.set_mode(m),
                          bg='#16213e', fg='white', selectcolor='#0f3460',
                          activebackground='#16213e', font=('Arial', 9)).pack(anchor=tk.W, padx=15)
        
        # Class selection
        class_frame = tk.LabelFrame(right_panel, text="[ Classes (1-9) ]", bg='#16213e', fg='#e94560',
                                    font=('Arial', 10, 'bold'))
        class_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        self.class_listbox = tk.Listbox(class_frame, bg='#0f3460', fg='white', font=('Consolas', 9),
                                        selectbackground='#e94560', selectforeground='white')
        self.class_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.class_listbox.bind('<<ListboxSelect>>', self.on_class_select)
        
        # Annotation list
        ann_frame = tk.LabelFrame(right_panel, text="[ Annotations ]", bg='#16213e', fg='#e94560',
                                 font=('Arial', 10, 'bold'))
        ann_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        self.ann_listbox = tk.Listbox(ann_frame, bg='#0f3460', fg='white', font=('Consolas', 8),
                                     selectbackground='#4ECDC4')
        self.ann_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # File list
        file_frame = tk.LabelFrame(right_panel, text="[ Images ]", bg='#16213e', fg='#e94560',
                                  font=('Arial', 10, 'bold'))
        file_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.files_listbox = tk.Listbox(file_frame, bg='#0f3460', fg='white', font=('Consolas', 8),
                                        height=6, selectbackground='#e94560')
        self.files_listbox.pack(fill=tk.X, padx=5, pady=5)
        self.files_listbox.bind('<<ListboxSelect>>', self.on_file_select)
        
        # Status bar
        status = tk.Frame(self.root, bg='#16213e', height=30)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = tk.Label(status, text="Ready", bg='#16213e', fg='white',
                                     anchor=tk.W, font=('Arial', 9))
        self.status_label.pack(fill=tk.X, padx=10, pady=5)
        
        # Keyboard shortcuts
        self.canvas.focus_set()
        self.canvas.bind("<Key>", self.on_key)
        
    def load_domain(self):
        config = DOMAINS[self.current_domain]
        
        self.class_listbox.delete(0, tk.END)
        
        if 'keypoints' in config:
            self.keypoint_names = config['keypoints']
            for i, kp in enumerate(self.keypoint_names[:30]):
                self.class_listbox.insert(tk.END, f"{i+1:2d}. {kp}")
        else:
            for i, cls in enumerate(config['classes'][:60]):
                self.class_listbox.insert(tk.END, f"{i+1:2d}. {cls}")
        
        self.status(f"Loaded: {config['name']} - {len(config['classes'])} classes")
        
    def on_domain_change(self, event=None):
        idx = self.domain_combo.current()
        domain_keys = list(DOMAINS.keys())
        self.current_domain = domain_keys[idx]
        self.annotations = []
        self.load_domain()
        self.redraw()
        self.update_ann_list()
        
    def set_mode(self, mode):
        self.mode = mode
        self.current_polygon = []
        self.status(f"Mode: {mode}")
        
    def on_class_select(self, event=None):
        selection = self.class_listbox.curselection()
        if selection:
            self.current_class = selection[0]
            
    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.image_files = [path]
            self.load_image(path)
            
    def open_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.image_files = sorted([
                os.path.join(folder, f) for f in os.listdir(folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ])
            if self.image_files:
                self.files_listbox.delete(0, tk.END)
                for f in self.image_files:
                    self.files_listbox.insert(tk.END, os.path.basename(f))
                self.current_index = 0
                self.load_image(self.image_files[0])
                
    def load_image(self, path):
        try:
            self.current_image_path = path
            self.annotations = []
            
            img = Image.open(path)
            self.original_size = img.size
            
            canvas_w = self.canvas.winfo_width() or 1050
            canvas_h = self.canvas.winfo_height() or 850
            
            ratio = min(canvas_w / img.width, canvas_h / img.height, 1.0)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            
            self.display_image = img.resize(new_size, Image.LANCZOS)
            self.scale = ratio
            
            self.tk_image = ImageTk.PhotoImage(self.display_image)
            
            self.offset_x = (canvas_w - new_size[0]) // 2
            self.offset_y = (canvas_h - new_size[1]) // 2
            
            self.canvas.delete("all")
            self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_image)
            
            self.status(f"Loaded: {os.path.basename(path)} ({img.width}x{img.height}) [{self.current_index+1}/{len(self.image_files)}]")
            self.update_ann_list()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    def on_click(self, event):
        x, y = event.x - self.offset_x, event.y - self.offset_y
        if x < 0 or y < 0:
            return
            
        if self.mode == 'keypoint':
            self.add_keypoint(x, y)
        elif self.mode == 'polygon':
            self.current_polygon.append((x, y))
            self.redraw()
        else:
            self.start_x, self.start_y = x, y
            self.drawing = True
            
    def on_drag(self, event):
        if self.drawing and self.mode == 'bbox':
            x, y = event.x - self.offset_x, event.y - self.offset_y
            self.canvas.delete("temp")
            self.canvas.create_rectangle(
                self.start_x + self.offset_x, self.start_y + self.offset_y,
                x + self.offset_x, y + self.offset_y,
                outline='red', width=2, tags="temp"
            )
            
    def on_release(self, event):
        if self.drawing and self.mode == 'bbox':
            self.drawing = False
            x, y = event.x - self.offset_x, event.y - self.offset_y
            
            x1, y1 = min(self.start_x, x), min(self.start_y, y)
            x2, y2 = max(self.start_x, x), max(self.start_y, y)
            
            if x2 - x1 > 5 and y2 - y1 > 5:
                self.annotations.append({
                    'type': 'bbox',
                    'class': self.current_class,
                    'class_name': self.get_class_name(self.current_class),
                    'points': [x1, y1, x2, y2]
                })
                self.redraw()
                self.update_ann_list()
                
    def on_double_click(self, event):
        if self.mode == 'polygon' and len(self.current_polygon) >= 3:
            self.annotations.append({
                'type': 'polygon',
                'class': self.current_class,
                'class_name': self.get_class_name(self.current_class),
                'points': self.current_polygon[:]
            })
            self.current_polygon = []
            self.redraw()
            self.update_ann_list()
            
    def add_keypoint(self, x, y):
        self.annotations.append({
            'type': 'keypoint',
            'class': self.current_class,
            'class_name': self.get_class_name(self.current_class),
            'points': [x, y]
        })
        self.redraw()
        self.update_ann_list()
        
    def get_class_name(self, idx):
        config = DOMAINS[self.current_domain]
        if 'keypoints' in config:
            return config['keypoints'][idx] if idx < len(config['keypoints']) else f"kp_{idx}"
        return config['classes'][idx] if idx < len(config['classes']) else f"class_{idx}"
        
    def redraw(self):
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_image)
        
        config = DOMAINS[self.current_domain]
        colors = config['color']
        
        for ann in self.annotations:
            color = colors[ann['class'] % len(colors)]
            
            if ann['type'] == 'bbox':
                x1, y1, x2, y2 = ann['points']
                self.canvas.create_rectangle(
                    x1 + self.offset_x, y1 + self.offset_y,
                    x2 + self.offset_x, y2 + self.offset_y,
                    outline=color, width=2, tags="ann"
                )
                self.canvas.create_text(
                    x1 + self.offset_x + 3, y1 + self.offset_y + 3,
                    text=ann['class_name'], fill=color, font=('Arial', 9, 'bold'), tags="ann"
                )
                
            elif ann['type'] == 'polygon':
                if len(ann['points']) > 1:
                    coords = []
                    for p in ann['points']:
                        coords.extend([p[0] + self.offset_x, p[1] + self.offset_y])
                    self.canvas.create_line(coords, fill=color, width=2, tags="ann")
                    for p in ann['points']:
                        self.canvas.create_oval(
                            p[0] + self.offset_x - 3, p[1] + self.offset_y - 3,
                            p[0] + self.offset_x + 3, p[1] + self.offset_y + 3,
                            fill=color, tags="ann"
                        )
                        
            elif ann['type'] == 'keypoint':
                x, y = ann['points']
                r = 5
                self.canvas.create_oval(
                    x + self.offset_x - r, y + self.offset_y - r,
                    x + self.offset_x + r, y + self.offset_y + r,
                    fill=color, tags="ann"
                )
                self.canvas.create_text(
                    x + self.offset_x + 8, y + self.offset_y - 5,
                    text=ann['class_name'], fill=color, font=('Arial', 7), tags="ann"
                )
        
        if self.current_polygon:
            coords = []
            for p in self.current_polygon:
                coords.extend([p[0] + self.offset_x, p[1] + self.offset_y])
            if len(coords) >= 4:
                self.canvas.create_line(coords, fill='white', width=2, dash=(4,4))
            for p in self.current_polygon:
                self.canvas.create_oval(
                    p[0] + self.offset_x - 3, p[1] + self.offset_y - 3,
                    p[0] + self.offset_x + 3, p[1] + self.offset_y + 3,
                    fill='white'
                )
                
    def update_ann_list(self):
        self.ann_listbox.delete(0, tk.END)
        for i, ann in enumerate(self.annotations):
            self.ann_listbox.insert(tk.END, f"{i+1}. [{ann['type']}] {ann['class_name']}")
            
    def on_file_select(self, event):
        selection = self.files_listbox.curselection()
        if selection:
            self.current_index = selection[0]
            self.load_image(self.image_files[self.current_index])
            
    def on_key(self, event):
        key = event.keysym
        
        if event.state & 0x4:
            if key == 's':
                self.save_annotations()
            elif key == 'z':
                if self.annotations:
                    self.annotations.pop()
                    self.redraw()
                    self.update_ann_list()
        elif key == 'Escape':
            self.current_polygon = []
            self.redraw()
        elif key.isdigit():
            idx = int(key) - 1
            if idx >= 0:
                self.current_class = idx
                self.class_listbox.selection_clear(0, tk.END)
                self.class_listbox.selection_set(idx)
        elif key == 'b':
            self.set_mode('bbox')
        elif key == 'p':
            self.set_mode('polygon')
        elif key == 'k':
            self.set_mode('keypoint')
        elif key in ['Left', 'Right']:
            self.navigate_image(key)
            
    def navigate_image(self, key):
        if not self.image_files:
            return
        delta = 1 if key == 'Right' else -1
        new_idx = self.current_index + delta
        if 0 <= new_idx < len(self.image_files):
            self.current_index = new_idx
            self.load_image(self.image_files[new_idx])
            self.files_listbox.selection_clear(0, tk.END)
            self.files_listbox.selection_set(new_idx)
            
    def save_annotations(self):
        if not self.current_image_path:
            return
            
        base = os.path.splitext(self.current_image_path)[0]
        
        yolo_path = base + '.txt'
        with open(yolo_path, 'w') as f:
            for ann in self.annotations:
                if ann['type'] == 'bbox':
                    x1, y1, x2, y2 = ann['points']
                    cx = ((x1 + x2) / 2) / self.original_size[0]
                    cy = ((y1 + y2) / 2) / self.original_size[1]
                    w = (x2 - x1) / self.original_size[0]
                    h = (y2 - y1) / self.original_size[1]
                    f.write(f"{ann['class']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    
        json_path = base + '.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'image': os.path.basename(self.current_image_path),
                'width': self.original_size[0],
                'height': self.original_size[1],
                'domain': self.current_domain,
                'annotations': self.annotations
            }, f, ensure_ascii=False, indent=2)
            
        self.status(f"Saved: {os.path.basename(yolo_path)} | {len(self.annotations)} annotations")
        
    def export_format(self, fmt):
        folder = filedialog.askdirectory(title=f"Export to {fmt.upper()}")
        if folder:
            messagebox.showinfo("Export", f"Export {fmt.upper()} configured for: {folder}")
            
    def status(self, msg):
        self.status_label.config(text=msg)


def main():
    root = tk.Tk()
    app = UniversalAnnotationTool(root)
    root.mainloop()


if __name__ == '__main__':
    main()
