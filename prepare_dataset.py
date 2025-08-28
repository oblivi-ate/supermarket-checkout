#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超市商品数据集准备脚本
用于收集、标注和预处理训练数据
"""

import os
import cv2
import json
import shutil
import random
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm

class DatasetPreparer:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        self.raw_dir = self.data_dir / "raw"
        
        # 创建目录结构
        self.create_directories()
        
        # 商品类别映射
        self.class_names = ['dairy', 'snacks', 'oil', 'water', 'bread']
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}
        
    def create_directories(self):
        """创建数据集目录结构"""
        dirs = [
            self.images_dir / "train",
            self.images_dir / "val", 
            self.images_dir / "test",
            self.labels_dir / "train",
            self.labels_dir / "val",
            self.labels_dir / "test",
            self.raw_dir / "collected",
            self.raw_dir / "annotated"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"✅ 数据集目录结构已创建: {self.data_dir}")
        
    def collect_images_from_camera(self, num_images=100, class_name="dairy"):
        """从摄像头收集图像数据"""
        if class_name not in self.class_names:
            print(f"❌ 错误: 类别 '{class_name}' 不在支持的类别中: {self.class_names}")
            return
            
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 无法打开摄像头")
            return
            
        collected_dir = self.raw_dir / "collected" / class_name
        collected_dir.mkdir(exist_ok=True)
        
        print(f"📸 开始收集 {class_name} 类别的图像")
        print("操作说明:")
        print("- 按 SPACE 键拍照")
        print("- 按 ESC 键退出")
        print(f"- 目标收集数量: {num_images}")
        
        collected_count = 0
        
        while collected_count < num_images:
            ret, frame = cap.read()
            if not ret:
                print("❌ 摄像头读取失败")
                break
                
            # 显示当前帧和收集进度
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Class: {class_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Collected: {collected_count}/{num_images}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to capture, ESC to exit", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Data Collection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # 空格键拍照
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{class_name}_{timestamp}.jpg"
                filepath = collected_dir / filename
                
                cv2.imwrite(str(filepath), frame)
                collected_count += 1
                print(f"✅ 已保存: {filename} ({collected_count}/{num_images})")
                
            elif key == 27:  # ESC键退出
                break
                
        cap.release()
        cv2.destroyAllWindows()
        print(f"📸 数据收集完成! 共收集 {collected_count} 张图像")
        
    def create_sample_annotations(self):
        """创建示例标注文件"""
        print("📝 创建示例标注文件...")
        
        # 为每个类别创建示例标注
        for class_name in self.class_names:
            collected_dir = self.raw_dir / "collected" / class_name
            if not collected_dir.exists():
                continue
                
            annotated_dir = self.raw_dir / "annotated" / class_name
            annotated_dir.mkdir(exist_ok=True)
            
            # 获取该类别的所有图像
            image_files = list(collected_dir.glob("*.jpg"))
            
            for img_file in image_files[:10]:  # 只处理前10张作为示例
                # 读取图像获取尺寸
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                    
                h, w = img.shape[:2]
                
                # 创建示例边界框（图像中心区域）
                # YOLO格式: class_id center_x center_y width height (归一化坐标)
                center_x = 0.5
                center_y = 0.5
                bbox_width = 0.6
                bbox_height = 0.6
                
                class_id = self.class_to_id[class_name]
                
                # 创建标注文件
                label_file = annotated_dir / f"{img_file.stem}.txt"
                with open(label_file, 'w') as f:
                    f.write(f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}\n")
                    
                # 复制图像到标注目录
                shutil.copy2(img_file, annotated_dir / img_file.name)
                
        print("✅ 示例标注文件创建完成")
        
    def split_dataset(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """将数据集分割为训练、验证和测试集"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须等于1"
        
        print("🔄 开始分割数据集...")
        
        all_images = []
        all_labels = []
        
        # 收集所有标注的图像和标签
        for class_name in self.class_names:
            annotated_dir = self.raw_dir / "annotated" / class_name
            if not annotated_dir.exists():
                continue
                
            image_files = list(annotated_dir.glob("*.jpg"))
            for img_file in image_files:
                label_file = annotated_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    all_images.append(img_file)
                    all_labels.append(label_file)
                    
        if not all_images:
            print("❌ 没有找到标注的图像数据")
            return
            
        # 随机打乱数据
        combined = list(zip(all_images, all_labels))
        random.shuffle(combined)
        all_images, all_labels = zip(*combined)
        
        total_count = len(all_images)
        train_count = int(total_count * train_ratio)
        val_count = int(total_count * val_ratio)
        
        # 分割数据
        train_images = all_images[:train_count]
        train_labels = all_labels[:train_count]
        
        val_images = all_images[train_count:train_count + val_count]
        val_labels = all_labels[train_count:train_count + val_count]
        
        test_images = all_images[train_count + val_count:]
        test_labels = all_labels[train_count + val_count:]
        
        # 复制文件到对应目录
        splits = [
            ("train", train_images, train_labels),
            ("val", val_images, val_labels),
            ("test", test_images, test_labels)
        ]
        
        for split_name, images, labels in splits:
            if not images:
                continue
                
            print(f"📁 处理 {split_name} 集: {len(images)} 个样本")
            
            for img_file, label_file in tqdm(zip(images, labels), desc=f"复制{split_name}数据"):
                # 复制图像
                dst_img = self.images_dir / split_name / img_file.name
                shutil.copy2(img_file, dst_img)
                
                # 复制标签
                dst_label = self.labels_dir / split_name / label_file.name
                shutil.copy2(label_file, dst_label)
                
        print(f"✅ 数据集分割完成!")
        print(f"   训练集: {len(train_images)} 个样本")
        print(f"   验证集: {len(val_images)} 个样本")
        print(f"   测试集: {len(test_images)} 个样本")
        
    def visualize_annotations(self, split="train", num_samples=5):
        """可视化标注数据"""
        images_dir = self.images_dir / split
        labels_dir = self.labels_dir / split
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"❌ {split} 数据集不存在")
            return
            
        image_files = list(images_dir.glob("*.jpg"))[:num_samples]
        
        print(f"🖼️ 可视化 {split} 集的标注数据...")
        
        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                continue
                
            # 读取图像
            img = cv2.imread(str(img_file))
            if img is None:
                continue
                
            h, w = img.shape[:2]
            
            # 读取标注
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            # 绘制边界框
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                    
                class_id = int(parts[0])
                center_x = float(parts[1]) * w
                center_y = float(parts[2]) * h
                bbox_w = float(parts[3]) * w
                bbox_h = float(parts[4]) * h
                
                # 计算边界框坐标
                x1 = int(center_x - bbox_w / 2)
                y1 = int(center_y - bbox_h / 2)
                x2 = int(center_x + bbox_w / 2)
                y2 = int(center_y + bbox_h / 2)
                
                # 绘制边界框和标签
                color = (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                class_name = self.class_names[class_id]
                cv2.putText(img, class_name, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                           
            # 显示图像
            cv2.imshow(f'Annotation - {img_file.name}', img)
            cv2.waitKey(0)
            
        cv2.destroyAllWindows()
        
    def generate_statistics(self):
        """生成数据集统计信息"""
        print("📊 生成数据集统计信息...")
        
        stats = {
            "total_images": 0,
            "total_annotations": 0,
            "class_distribution": {name: 0 for name in self.class_names},
            "splits": {}
        }
        
        for split in ["train", "val", "test"]:
            images_dir = self.images_dir / split
            labels_dir = self.labels_dir / split
            
            if not images_dir.exists():
                continue
                
            image_files = list(images_dir.glob("*.jpg"))
            split_stats = {
                "images": len(image_files),
                "annotations": 0,
                "class_count": {name: 0 for name in self.class_names}
            }
            
            for img_file in image_files:
                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            class_name = self.class_names[class_id]
                            split_stats["class_count"][class_name] += 1
                            split_stats["annotations"] += 1
                            
            stats["splits"][split] = split_stats
            stats["total_images"] += split_stats["images"]
            stats["total_annotations"] += split_stats["annotations"]
            
            for class_name, count in split_stats["class_count"].items():
                stats["class_distribution"][class_name] += count
                
        # 打印统计信息
        print("\n" + "="*50)
        print("📊 数据集统计信息")
        print("="*50)
        print(f"总图像数量: {stats['total_images']}")
        print(f"总标注数量: {stats['total_annotations']}")
        
        print("\n📈 类别分布:")
        for class_name, count in stats["class_distribution"].items():
            percentage = (count / stats['total_annotations'] * 100) if stats['total_annotations'] > 0 else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
            
        print("\n📁 数据集分割:")
        for split, split_stats in stats["splits"].items():
            print(f"  {split}: {split_stats['images']} 图像, {split_stats['annotations']} 标注")
            
        # 保存统计信息到文件
        stats_file = self.data_dir / "dataset_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
            
        print(f"\n💾 统计信息已保存到: {stats_file}")
        
    def create_annotation_template(self):
        """创建标注模板和说明"""
        template_dir = self.data_dir / "annotation_templates"
        template_dir.mkdir(exist_ok=True)
        
        # 创建标注说明文件
        readme_content = """
# 数据标注说明

## YOLO格式标注规范

每个图像对应一个同名的.txt标注文件，格式如下：
```
class_id center_x center_y width height
```

### 参数说明：
- `class_id`: 类别ID (0-4)
- `center_x`: 边界框中心点x坐标 (归一化，0-1)
- `center_y`: 边界框中心点y坐标 (归一化，0-1) 
- `width`: 边界框宽度 (归一化，0-1)
- `height`: 边界框高度 (归一化，0-1)

### 类别映射：
0: dairy (乳制品)
1: snacks (零食)
2: oil (食用油)
3: water (水)
4: bread (面包)

### 标注示例：
```
0 0.5 0.5 0.6 0.4
```
表示：乳制品类别，边界框中心在图像中心，宽度占图像60%，高度占图像40%

### 标注工具推荐：
1. labelImg: https://github.com/tzutalin/labelImg
2. CVAT: https://github.com/openvinotoolkit/cvat
3. Roboflow: https://roboflow.com/

### 标注质量要求：
1. 边界框应紧贴目标物体
2. 确保类别标签正确
3. 避免遗漏小目标
4. 处理遮挡情况时标注可见部分
"""
        
        readme_file = template_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
            
        print(f"📝 标注说明已创建: {readme_file}")

def main():
    parser = argparse.ArgumentParser(description="超市商品数据集准备工具")
    parser.add_argument('--action', type=str, required=True,
                       choices=['collect', 'annotate', 'split', 'visualize', 'stats', 'template'],
                       help='执行的操作')
    parser.add_argument('--class-name', type=str, default='dairy',
                       choices=['dairy', 'snacks', 'oil', 'water', 'bread'],
                       help='商品类别名称')
    parser.add_argument('--num-images', type=int, default=100,
                       help='收集图像数量')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='数据集分割')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='可视化样本数量')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='数据目录路径')
    
    args = parser.parse_args()
    
    # 创建数据集准备器
    preparer = DatasetPreparer(args.data_dir)
    
    if args.action == 'collect':
        preparer.collect_images_from_camera(args.num_images, args.class_name)
    elif args.action == 'annotate':
        preparer.create_sample_annotations()
    elif args.action == 'split':
        preparer.split_dataset()
    elif args.action == 'visualize':
        preparer.visualize_annotations(args.split, args.num_samples)
    elif args.action == 'stats':
        preparer.generate_statistics()
    elif args.action == 'template':
        preparer.create_annotation_template()
        
if __name__ == "__main__":
    main()