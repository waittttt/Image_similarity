import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel


# === 1. 载入 ResNet 模型（预训练） ===
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # 去掉最后分类层
model.eval()  # 设置为评估模式（不训练）

# === 2. 定义图像预处理（转换为 ResNet 兼容格式） ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === 3. 提取特征向量 ===
def extract_features(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 转换格式
    img = transform(img).unsqueeze(0)  # 加 batch 维度
    with torch.no_grad():
        features = model(img)
    return features.flatten().numpy()


# === 4. 计算余弦相似度 ===
def deep_similarity(img1, img2):
    feat1 = extract_features(img1)
    feat2 = extract_features(img2)
    return cosine_similarity([feat1], [feat2])[0][0]  # 返回相似度分数


# === 5. GUI 设计（PyQt5） ===
class ImageSimilarityTool(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Deep Learning Image Similarity Tool')
        self.layout = QVBoxLayout()

        # 选择文件夹按钮
        self.load_button = QPushButton('Load Image Folder')
        self.layout.addWidget(self.load_button)

        # 结果显示
        self.similarity_label = QLabel('Similarity Results: ')
        self.layout.addWidget(self.similarity_label)

        # 绑定按钮
        self.load_button.clicked.connect(self.load_images)

        self.setLayout(self.layout)

    def load_images(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            images = self.load_image_files(folder)
            similarities = self.compute_similarities(images)
            self.display_similarities(similarities)

    def load_image_files(self, folder):
        images = []
        for filename in os.listdir(folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(folder, filename))
                images.append((filename, img))
        return images

    def compute_similarities(self, images):
        similarities = []
        features = {name: extract_features(img) for name, img in images}  # 提取所有图片特征

        for i, (name1, feat1) in enumerate(features.items()):
            for j, (name2, feat2) in enumerate(features.items()):
                if i < j:
                    sim_score = cosine_similarity([feat1], [feat2])[0][0]  # 计算相似度
                    similarities.append((name1, name2, sim_score))

        return sorted(similarities, key=lambda x: x[2], reverse=True)  # 排序（相似度高的在前）

    def display_similarities(self, similarities):
        result = ""
        for sim in similarities[:100]:  # 只显示前 100 个最相似的
            result += f"{sim[0]} vs {sim[1]}: Similarity Score = {sim[2]:.4f}\n"
        self.similarity_label.setText(result)


# === 6. 运行程序 ===
if __name__ == '__main__':
    app = QApplication([])
    tool = ImageSimilarityTool()
    tool.show()
    app.exec_()