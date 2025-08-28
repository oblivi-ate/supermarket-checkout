#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超市商品识别模型训练启动脚本
提供简化的训练流程和参数配置
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import yaml

def check_environment():
    """检查训练环境"""
    print("🔍 检查训练环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print("❌ Python版本过低，需要Python 3.7+")
        return False
        
    # 检查必要文件
    required_files = [
        "train.py",
        "data/supermarket.yaml", 
        "data/hyp.scratch.p5.yaml",
        "weights/yolov7.pt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            
    if missing_files:
        print("❌ 缺少必要文件:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
        
    # 检查数据集
    data_dirs = [
        "data/images/train",
        "data/images/val", 
        "data/labels/train",
        "data/labels/val"
    ]
    
    empty_dirs = []
    for dir_path in data_dirs:
        dir_obj = Path(dir_path)
        if not dir_obj.exists() or not any(dir_obj.iterdir()):
            empty_dirs.append(dir_path)
            
    if empty_dirs:
        print("⚠️ 以下数据目录为空:")
        for dir_path in empty_dirs:
            print(f"   - {dir_path}")
        print("请先使用 prepare_dataset.py 准备训练数据")
        return False
        
    print("✅ 环境检查通过")
    return True

def install_requirements():
    """安装依赖包"""
    print("📦 检查并安装依赖包...")
    
    try:
        import torch
        import torchvision
        import cv2
        import yaml
        import tqdm
        import matplotlib
        import seaborn
        import pandas
        import scipy
        print("✅ 所有依赖包已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖包: {e}")
        print("正在安装依赖包...")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ 依赖包安装完成")
            return True
        except subprocess.CalledProcessError:
            print("❌ 依赖包安装失败")
            return False

def create_training_config(args):
    """创建训练配置"""
    config = {
        'weights': args.weights,
        'cfg': args.cfg,
        'data': args.data,
        'hyp': args.hyp,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'img_size': args.img_size,
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': args.name
    }
    
    return config

def start_training(config):
    """启动训练"""
    print("🚀 开始训练...")
    print(f"配置信息:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # 构建训练命令
    cmd = [
        sys.executable, "train.py",
        "--weights", str(config['weights']),
        "--cfg", str(config['cfg']),
        "--data", str(config['data']),
        "--hyp", str(config['hyp']),
        "--epochs", str(config['epochs']),
        "--batch-size", str(config['batch_size']),
        "--img-size", str(config['img_size'][0]), str(config['img_size'][1]),
        "--device", str(config['device']),
        "--workers", str(config['workers']),
        "--project", str(config['project']),
        "--name", str(config['name'])
    ]
    
    # 添加可选参数
    if config.get('cache_images'):
        cmd.append("--cache-images")
    if config.get('multi_scale'):
        cmd.append("--multi-scale")
    if config.get('adam'):
        cmd.append("--adam")
        
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        # 启动训练进程
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 universal_newlines=True, bufsize=1)
        
        # 实时输出训练日志
        for line in process.stdout:
            print(line.rstrip())
            
        process.wait()
        
        if process.returncode == 0:
            print("\n🎉 训练完成!")
            print(f"训练结果保存在: {config['project']}/{config['name']}")
        else:
            print(f"\n❌ 训练失败，退出码: {process.returncode}")
            
    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")
        process.terminate()
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")

def main():
    parser = argparse.ArgumentParser(description="超市商品识别模型训练启动器")
    
    # 基础参数
    parser.add_argument('--weights', type=str, default='weights/yolov7.pt', 
                       help='预训练权重路径')
    parser.add_argument('--cfg', type=str, default='', 
                       help='模型配置文件路径')
    parser.add_argument('--data', type=str, default='data/supermarket.yaml', 
                       help='数据集配置文件路径')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', 
                       help='超参数配置文件路径')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, 
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, 
                       help='批次大小')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], 
                       help='输入图像尺寸 [训练, 测试]')
    parser.add_argument('--device', default='', 
                       help='训练设备 (cpu, 0, 1, 2, 3 等)')
    parser.add_argument('--workers', type=int, default=8, 
                       help='数据加载器工作进程数')
    
    # 输出参数
    parser.add_argument('--project', default='runs/train', 
                       help='项目保存路径')
    parser.add_argument('--name', default='supermarket_model', 
                       help='实验名称')
    
    # 可选参数
    parser.add_argument('--cache-images', action='store_true', 
                       help='缓存图像以加速训练')
    parser.add_argument('--multi-scale', action='store_true', 
                       help='多尺度训练')
    parser.add_argument('--adam', action='store_true', 
                       help='使用Adam优化器')
    
    # 控制参数
    parser.add_argument('--skip-check', action='store_true', 
                       help='跳过环境检查')
    parser.add_argument('--install-deps', action='store_true', 
                       help='自动安装依赖')
    
    args = parser.parse_args()
    
    print("🏪 超市商品识别模型训练启动器")
    print("=" * 50)
    
    # 环境检查
    if not args.skip_check:
        if not check_environment():
            print("\n❌ 环境检查失败，请解决上述问题后重试")
            return
    
    # 安装依赖
    if args.install_deps:
        if not install_requirements():
            print("\n❌ 依赖安装失败")
            return
    
    # 创建训练配置
    config = create_training_config(args)
    config['cache_images'] = args.cache_images
    config['multi_scale'] = args.multi_scale
    config['adam'] = args.adam
    
    # 开始训练
    start_training(config)

if __name__ == "__main__":
    main()