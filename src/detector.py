import cv2
import torch
import os
import sys
import numpy as np
import types

# ===== 猴子补丁：修复 Git 和下载问题 =====
# 创建虚拟的 google_utils 模块
google_utils = types.ModuleType('google_utils')
google_utils.attempt_download = lambda file: file  # 直接返回文件路径
sys.modules['utils.google_utils'] = google_utils

# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 从本地 models 导入
from models.experimental import attempt_load
from utils.general import non_max_suppression

class ProductDetector:
    def __init__(self, weights_path='weights/yolov7.pt', conf_thres=0.7):
        # 获取项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)  # 上一级目录
        
        # 构建绝对路径
        abs_weights_path = os.path.abspath(
            os.path.join(project_root, weights_path)
        )
        
        # 检查权重文件是否存在
        if not os.path.exists(abs_weights_path):
            raise FileNotFoundError(f"权重文件不存在: {abs_weights_path}")
            # 或者自动下载
            # self.download_weights(abs_weights_path)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        print(f"加载模型权重: {abs_weights_path}")
        
        try:
            # 直接加载模型
            self.model = attempt_load(abs_weights_path, map_location=self.device)
        except Exception as e:
            print(f"加载模型失败: {e}")
            # 创建简化模型作为后备
            self.model = self.create_fallback_model()
        
        self.conf_thres = conf_thres
        self.classes = ['dairy', 'snacks', 'oil', 'water', 'bread']  # 商品类别标签
        
        # 启用半精度加速（如果可用）
        if self.device.type != 'cpu':
            self.model.half()
            print("启用半精度推理")
    
    def download_weights(self, path):
        """手动下载权重文件（备用）"""
        import urllib.request
        print("下载权重文件中...")
        url = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"
        urllib.request.urlretrieve(url, path)
        print(f"权重文件已下载到: {path}")
    
    def create_fallback_model(self):
        """创建简化模型作为后备"""
        print("创建后备简化模型")
        import torch.nn as nn
        
        class FallbackModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.fc = nn.Linear(32 * 160 * 160, 5)  # 5个类别
            
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        return FallbackModel().to(self.device).eval()

    def detect(self, frame):
        # 预处理
        img = self.preprocess(frame)
        
        # 推理
        with torch.no_grad():
            try:
                pred = self.model(img)[0]
            except Exception as e:
                print(f"推理错误: {e}")
                return []
        
        # 后处理
        detections = self.postprocess(pred, img.shape)
        return detections

    def preprocess(self, frame):
        # 调整大小
        img = cv2.resize(frame, (640, 640))
        
        # 转换颜色空间和通道顺序
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        
        # 转换为Tensor
        img = torch.from_numpy(img).to(self.device)
        
        # 转换为浮点并归一化
        img = img.half() if self.device.type != 'cpu' else img.float()
        img = img / 255.0
        
        # 添加批次维度
        return img.unsqueeze(0)

    def postprocess(self, pred, img_shape):
        # 应用非极大值抑制
        pred = non_max_suppression(pred, self.conf_thres, 0.45)
        
        detections = []
        for i, det in enumerate(pred):
            if det is not None and len(det):
                for *xyxy, conf, cls in reversed(det):
                    class_idx = int(cls)
                    class_name = self.classes[class_idx] if class_idx < len(self.classes) else f"class_{class_idx}"
                    
                    detections.append({
                        'class': class_name,
                        'confidence': conf.item(),
                        'bbox': [coord.item() for coord in xyxy]
                    })
        return detections