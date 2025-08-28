#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超市自助收银系统 - 环境设置脚本

这个脚本帮助用户快速设置项目环境，包括:
- 创建虚拟环境
- 安装依赖包
- 下载预训练模型
- 初始化数据目录
- 验证环境配置
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path
import zipfile
import shutil

class EnvironmentSetup:
    """环境设置类"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / 'venv'
        self.weights_path = self.project_root / 'weights'
        self.data_path = self.project_root / 'data'
        
    def print_step(self, step, message):
        """打印步骤信息"""
        print(f"\n[{step}] {message}")
        print("-" * 50)
    
    def run_command(self, command, check=True):
        """运行命令"""
        try:
            result = subprocess.run(command, shell=True, check=check, 
                                  capture_output=True, text=True)
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.CalledProcessError as e:
            return False, e.stdout, e.stderr
    
    def check_python_version(self):
        """检查Python版本"""
        self.print_step("1/7", "检查Python版本")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 7):
            print(f"❌ Python版本过低: {version.major}.{version.minor}")
            print("请安装Python 3.7或更高版本")
            return False
        
        print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
        return True
    
    def create_virtual_environment(self):
        """创建虚拟环境"""
        self.print_step("2/7", "创建虚拟环境")
        
        if self.venv_path.exists():
            print("⚠️  虚拟环境已存在，跳过创建")
            return True
        
        success, stdout, stderr = self.run_command(f"python -m venv {self.venv_path}")
        if success:
            print("✅ 虚拟环境创建成功")
            return True
        else:
            print(f"❌ 虚拟环境创建失败: {stderr}")
            return False
    
    def install_dependencies(self):
        """安装依赖包"""
        self.print_step("3/7", "安装依赖包")
        
        # 激活虚拟环境的pip路径
        if os.name == 'nt':  # Windows
            pip_path = self.venv_path / 'Scripts' / 'pip.exe'
        else:  # Linux/Mac
            pip_path = self.venv_path / 'bin' / 'pip'
        
        if not pip_path.exists():
            print("⚠️  使用系统pip安装依赖")
            pip_path = 'pip'
        
        # 升级pip
        print("📦 升级pip...")
        success, _, _ = self.run_command(f"{pip_path} install --upgrade pip")
        
        # 安装依赖
        requirements_file = self.project_root / 'requirements.txt'
        if requirements_file.exists():
            print("📦 安装项目依赖...")
            success, stdout, stderr = self.run_command(
                f"{pip_path} install -r {requirements_file}"
            )
            if success:
                print("✅ 依赖包安装成功")
                return True
            else:
                print(f"❌ 依赖包安装失败: {stderr}")
                return False
        else:
            print("⚠️  未找到requirements.txt文件")
            return False
    
    def download_pretrained_model(self):
        """下载预训练模型"""
        self.print_step("4/7", "下载预训练模型")
        
        # 创建weights目录
        self.weights_path.mkdir(exist_ok=True)
        
        model_file = self.weights_path / 'yolov7.pt'
        if model_file.exists():
            print("✅ 预训练模型已存在")
            return True
        
        # YOLOv7模型下载链接
        model_url = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"
        
        try:
            print("📥 正在下载YOLOv7模型... (约75MB)")
            print("这可能需要几分钟时间，请耐心等待...")
            
            # 下载模型
            urllib.request.urlretrieve(model_url, model_file)
            
            if model_file.exists() and model_file.stat().st_size > 1000000:  # 至少1MB
                print("✅ 预训练模型下载成功")
                return True
            else:
                print("❌ 模型文件下载不完整")
                return False
                
        except Exception as e:
            print(f"❌ 模型下载失败: {e}")
            print("请手动下载模型文件到 weights/yolov7.pt")
            print(f"下载链接: {model_url}")
            return False
    
    def setup_data_directories(self):
        """设置数据目录"""
        self.print_step("5/7", "设置数据目录")
        
        # 创建数据目录结构
        directories = [
            self.data_path / 'raw' / 'collected',
            self.data_path / 'raw' / 'annotated',
            self.data_path / 'train' / 'images',
            self.data_path / 'train' / 'labels',
            self.data_path / 'val' / 'images',
            self.data_path / 'val' / 'labels',
            self.data_path / 'test' / 'images',
            self.data_path / 'test' / 'labels',
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("✅ 数据目录结构创建完成")
        
        # 创建.gitkeep文件保持目录结构
        for directory in directories:
            gitkeep_file = directory / '.gitkeep'
            if not gitkeep_file.exists():
                gitkeep_file.touch()
        
        return True
    
    def verify_installation(self):
        """验证安装"""
        self.print_step("6/7", "验证安装")
        
        # 检查关键文件
        required_files = [
            'src/main.py',
            'src/detector.py',
            'src/checkout_system.py',
            'train.py',
            'demo.py',
            'data/supermarket.yaml',
            'data/hyp.scratch.p5.yaml'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            print("❌ 缺少以下文件:")
            for file_path in missing_files:
                print(f"   - {file_path}")
            return False
        
        # 测试Python导入
        test_imports = [
            'import cv2',
            'import torch',
            'import numpy as np',
            'import yaml'
        ]
        
        # 获取Python可执行文件路径
        if os.name == 'nt':  # Windows
            python_path = self.venv_path / 'Scripts' / 'python.exe'
        else:  # Linux/Mac
            python_path = self.venv_path / 'bin' / 'python'
        
        if not python_path.exists():
            python_path = 'python'
        
        failed_imports = []
        for import_cmd in test_imports:
            success, _, stderr = self.run_command(
                f"{python_path} -c \"{import_cmd}\"", check=False
            )
            if not success:
                failed_imports.append(import_cmd.split()[-1])
        
        if failed_imports:
            print("❌ 以下包导入失败:")
            for package in failed_imports:
                print(f"   - {package}")
            return False
        
        print("✅ 所有组件验证通过")
        return True
    
    def create_shortcuts(self):
        """创建快捷方式"""
        self.print_step("7/7", "创建快捷方式")
        
        # 创建启动脚本
        if os.name == 'nt':  # Windows
            # 已经有run_demo.bat了
            print("✅ Windows批处理文件已存在")
        else:  # Linux/Mac
            # 创建shell脚本
            shell_script = self.project_root / 'run_demo.sh'
            with open(shell_script, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write("echo '🛒 AI超市自助收银系统 - 演示启动器'\n")
                f.write("echo '========================================'\n")
                f.write("\n")
                f.write("# 激活虚拟环境\n")
                f.write("if [ -f 'venv/bin/activate' ]; then\n")
                f.write("    source venv/bin/activate\n")
                f.write("    echo '🔄 虚拟环境已激活'\n")
                f.write("fi\n")
                f.write("\n")
                f.write("# 启动演示\n")
                f.write("echo '🚀 启动演示系统...'\n")
                f.write("python demo.py\n")
            
            # 设置执行权限
            os.chmod(shell_script, 0o755)
            print("✅ Shell脚本创建完成")
        
        return True
    
    def run_setup(self):
        """运行完整设置流程"""
        print("🛒 AI超市自助收银系统 - 环境设置")
        print("=" * 60)
        
        steps = [
            self.check_python_version,
            self.create_virtual_environment,
            self.install_dependencies,
            self.download_pretrained_model,
            self.setup_data_directories,
            self.verify_installation,
            self.create_shortcuts
        ]
        
        for i, step in enumerate(steps, 1):
            if not step():
                print(f"\n❌ 设置失败，停止在步骤 {i}")
                return False
        
        print("\n" + "=" * 60)
        print("🎉 环境设置完成！")
        print("\n📋 接下来你可以:")
        print("1. 运行演示系统: python demo.py")
        print("2. 收集训练数据: python prepare_dataset.py --action collect")
        print("3. 训练自定义模型: python start_training.py")
        print("4. 启动完整系统: python -m src.main")
        print("\n📖 详细说明请查看 README.md 和 TRAINING_GUIDE.md")
        
        return True

def main():
    """主函数"""
    setup = EnvironmentSetup()
    
    try:
        success = setup.run_setup()
        if not success:
            print("\n💡 如果遇到问题，请检查:")
            print("1. Python版本是否为3.7+")
            print("2. 网络连接是否正常")
            print("3. 是否有足够的磁盘空间")
            print("4. 是否有管理员权限（如果需要）")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️  用户中断设置过程")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 设置过程中出现错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()