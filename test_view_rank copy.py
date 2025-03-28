import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel,
    QListWidget, QListWidgetItem, QLineEdit, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


# === 1. 载入 ResNet 模型 ===
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# === 2. 图像预处理 ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === 3. 提取特征 ===
def extract_features(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model(img)
    return features.flatten().numpy()


# === 4. GUI 设计 ===
class ImageSimilarityTool(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Similarity Tool')
        self.setGeometry(100, 100, 900, 600)

        self.layout = QHBoxLayout()
        self.left_panel = QVBoxLayout()
        self.right_panel = QVBoxLayout()

        # === 1. 加载按钮 ===
        self.load_button = QPushButton('Load Image Folder')
        self.left_panel.addWidget(self.load_button)

        # === 2. 相似度筛选框 ===
        filter_layout = QHBoxLayout()
        self.min_similarity_input = QLineEdit(self)
        self.min_similarity_input.setPlaceholderText("Min similarity (e.g., 0.5)")
        self.max_similarity_input = QLineEdit(self)
        self.max_similarity_input.setPlaceholderText("Max similarity (e.g., 0.6)")
        self.filter_button = QPushButton("Filter")
        self.filter_result_label = QLabel("Please load the dataset.")
        self.left_panel.addWidget(self.filter_result_label)
        filter_layout.addWidget(self.min_similarity_input)
        filter_layout.addWidget(self.max_similarity_input)
        filter_layout.addWidget(self.filter_button)
        self.left_panel.addLayout(filter_layout)

        # === 3. 排序按钮 ===
        self.sort_button = QPushButton("Sort Order: High → Low")
        self.left_panel.addWidget(self.sort_button)

        # === 4. 结果列表 ===
        self.result_list = QListWidget()
        self.left_panel.addWidget(self.result_list)

        # === 5. 显示图片 ===
        self.image_label_1 = QLabel()
        self.image_label_1.setAlignment(Qt.AlignCenter)
        self.image_name_label_1 = QLabel("")
        self.image_name_label_1.setAlignment(Qt.AlignCenter)

        self.image_label_2 = QLabel()
        self.image_label_2.setAlignment(Qt.AlignCenter)
        self.image_name_label_2 = QLabel("")
        self.image_name_label_2.setAlignment(Qt.AlignCenter)

        self.right_panel.addWidget(self.image_label_1)
        self.right_panel.addWidget(self.image_name_label_1)
        self.right_panel.addWidget(self.image_label_2)
        self.right_panel.addWidget(self.image_name_label_2)

        # === 6. 绑定事件 ===
        self.load_button.clicked.connect(self.load_images)
        self.result_list.itemClicked.connect(self.display_selected_images)
        self.filter_button.clicked.connect(self.filter_results)
        self.sort_button.clicked.connect(self.toggle_sort_order)

        # === 7. 设定默认排序方式（高到低） ===
        self.sort_descending = True  

        # === 8. 布局 ===
        self.layout.addLayout(self.left_panel)
        self.layout.addLayout(self.right_panel)
        self.setLayout(self.layout)

        self.image_folder = None
        self.all_similarities = []

    def load_images(self):
        self.image_folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if self.image_folder:
            images = self.load_image_files(self.image_folder)
            self.all_similarities = self.compute_similarities(images)
            self.display_similarities()

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
        features = {name: extract_features(img) for name, img, _ in images}

        for i, (name1, feat1) in enumerate(features.items()):
            for j, (name2, feat2) in enumerate(features.items()):
                if i < j:
                    sim_score = cosine_similarity([feat1], [feat2])[0][0]
                    similarities.append((name1, name2, sim_score))

        return similarities

    def display_similarities(self):
        """ 根据当前排序方式显示相似度列表 """
        self.result_list.clear()
        sorted_similarities = sorted(
            self.all_similarities,
            key=lambda x: x[2],
            reverse=self.sort_descending  # 根据用户选择的排序方式
        )

        for sim in sorted_similarities[:100]:  
            item = QListWidgetItem(f"{sim[0]} vs {sim[1]}: Similarity = {sim[2]:.4f}")
            item.setData(Qt.UserRole, (sim[0], sim[1], sim[2]))  
            self.result_list.addItem(item)
        if len(self.all_similarities)>=100:
            self.filter_result_label.setText(f"Filtered results: {len(self.all_similarities)} pairs found.Here are 100 pairs of results displayed.")
        else:
            self.filter_result_label.setText(f"Filtered results: {len(self.all_similarities)} pairs found.Here are {len(self.all_similarities)} pairs of results displayed.")

    def display_selected_images(self, item):
        if not self.image_folder:
            return

        image1_name, image2_name, _ = item.data(Qt.UserRole)  

        image1_path = os.path.join(self.image_folder, image1_name)
        image2_path = os.path.join(self.image_folder, image2_name)

        self.show_image(self.image_label_1, image1_path)
        self.show_image(self.image_label_2, image2_path)

        self.image_name_label_1.setText(image1_name)
        self.image_name_label_2.setText(image2_name)

    def show_image(self, label, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (200, 200))
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        label.setPixmap(pixmap)

    def filter_results(self):
        """ 筛选符合范围的相似度对 """
        try:
            min_sim = float(self.min_similarity_input.text())
            max_sim = float(self.max_similarity_input.text())
        except ValueError:
            return

        self.all_similarities = [
            sim for sim in self.compute_similarities(self.load_image_files(self.image_folder))
            if min_sim <= sim[2] <= max_sim
        ]
        self.display_similarities()

    def toggle_sort_order(self):
        """ 切换排序方式 """
        self.sort_descending = not self.sort_descending
        order_text = "High → Low" if self.sort_descending else "Low → High"
        self.sort_button.setText(f"Sort Order: {order_text}")
        self.display_similarities()


# === 运行程序 ===
if __name__ == '__main__':
    app = QApplication([])
    tool = ImageSimilarityTool()
    tool.show()
    app.exec_()
