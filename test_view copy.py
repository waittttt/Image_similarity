import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QListWidget, QListWidgetItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


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
        self.setGeometry(100, 100, 800, 600)  # 窗口大小

        self.layout = QVBoxLayout()

        # 选择文件夹按钮
        self.load_button = QPushButton('Load Image Folder')
        self.layout.addWidget(self.load_button)

        # 结果显示列表
        self.result_list = QListWidget()
        self.layout.addWidget(self.result_list)

        # 图片显示区域
        self.image_label_1 = QLabel("Image 1")
        self.image_label_1.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label_1)
        self.image_name_label_1 = QLabel("")  # **新增文件名显示**
        self.image_name_label_1.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_name_label_1)

        self.image_label_2 = QLabel("Image 2")
        self.image_label_2.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label_2)
        self.image_name_label_2 = QLabel("")  # **新增文件名显示**
        self.image_name_label_2.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_name_label_2)

        # 绑定按钮
        self.load_button.clicked.connect(self.load_images)
        self.result_list.itemClicked.connect(self.display_selected_images)  # 绑定点击事件

        self.setLayout(self.layout)
        self.image_folder = None  # 存储当前文件夹路径

    def load_images(self):
        self.image_folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if self.image_folder:
            images = self.load_image_files(self.image_folder)
            similarities = self.compute_similarities(images)
            self.display_similarities(similarities)

    def load_image_files(self, folder):
        images = []
        for filename in os.listdir(folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append((filename, img, img_path))
        return images

    def compute_similarities(self, images):
        similarities = []
        features = {name: extract_features(img) for name, img, _ in images}  # 提取所有图片特征

        for i, (name1, feat1) in enumerate(features.items()):
            for j, (name2, feat2) in enumerate(features.items()):
                if i < j:
                    sim_score = cosine_similarity([feat1], [feat2])[0][0]  # 计算相似度
                    similarities.append((name1, name2, sim_score))

        return sorted(similarities, key=lambda x: x[2], reverse=True)  # 排序（相似度高的在前）

    def display_similarities(self, similarities):
        self.result_list.clear()  # 清空列表
        for sim in similarities[:100]:  # 只显示前 100 个最相似的
            item = QListWidgetItem(f"{sim[0]} vs {sim[1]}: Similarity Score = {sim[2]:.4f}")
            item.setData(Qt.UserRole, (sim[0], sim[1]))  # 存储图片文件名
            self.result_list.addItem(item)

    def display_selected_images(self, item):
        if not self.image_folder:
            return
        
        # 获取用户点击的图片名
        image1_name, image2_name = item.data(Qt.UserRole)
        
        image1_path = os.path.join(self.image_folder, image1_name)
        image2_path = os.path.join(self.image_folder, image2_name)

        # 调试输出路径
        print(f"Displaying images: {image1_path}, {image2_path}")

        # 显示图片
        self.show_image(self.image_label_1, image1_path)
        self.show_image(self.image_label_2, image2_path)

        self.image_name_label_1.setText(image1_name)
        self.image_name_label_2.setText(image2_name)

    def show_image(self, label, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return
        
        # 转换 BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (200, 200))  # 调整大小
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        label.setPixmap(pixmap)  # 设置图片


# === 6. 运行程序 ===
if __name__ == '__main__':
    app = QApplication([])
    tool = ImageSimilarityTool()
    tool.show()
    app.exec_()
