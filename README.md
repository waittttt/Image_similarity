# 图像相似性检查工具

## 1. 功能说明

该工具通过深度学习模型（ResNet-18）提取图像特征，计算图像之间的相似度。用户可以查看图像对、筛选相似度结果，进行删除和重命名等操作。

## 2. 如何运行工具

### 1. 下载项目文件
1. 下载工具的源代码并解压到本地目录。

### 2. 安装依赖
确保你已安装以下 Python 库：
- `torch`
- `torchvision`
- `scikit-learn`
- `PyQt5`
- `opencv-python`
- `Pillow`

你可以使用以下命令安装依赖：
```bash
pip install -r requirements.txt
