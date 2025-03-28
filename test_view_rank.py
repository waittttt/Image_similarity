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
    QListWidget, QListWidgetItem, QLineEdit, QHBoxLayout, QInputDialog, QMessageBox
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

        # === 5. 删除按钮 ===
        self.delete_button = QPushButton("Delete Selected Image")
        self.right_panel.addWidget(self.delete_button)

        # === 6. 重命名按钮 ===
        self.rename_button = QPushButton("Rename Selected Image")
        self.right_panel.addWidget(self.rename_button)

        # === 7. 显示图片 ===
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

        # === 8. 绑定事件 ===
        self.load_button.clicked.connect(self.load_images)
        self.result_list.itemClicked.connect(self.display_selected_images)
        self.filter_button.clicked.connect(self.filter_results)
        self.sort_button.clicked.connect(self.toggle_sort_order)
        self.delete_button.clicked.connect(self.delete_image)  # 绑定删除事件
        self.rename_button.clicked.connect(self.rename_image)  # 绑定重命名事件

        # === 9. 设定默认排序方式（高到低） ===
        self.sort_descending = True  

        # === 10. 布局 ===
        self.layout.addLayout(self.left_panel)
        self.layout.addLayout(self.right_panel)
        self.setLayout(self.layout)

        self.image_folder = None
        self.all_similarities = []

        self.selected_image_index = None  # None 表示没有选中的图片
        self.selected_images = None      # 记录选中的图片对

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
        if len(self.all_similarities) >= 100:
            self.filter_result_label.setText(f"Filtered results: {len(self.all_similarities)} pairs found. Here are 100 pairs of results displayed.")
        else:
            self.filter_result_label.setText(f"Filtered results: {len(self.all_similarities)} pairs found. Here are {len(self.all_similarities)} pairs of results displayed.")

    def display_selected_images(self, item):
        if not self.image_folder:
            return

        image1_name, image2_name, _ = item.data(Qt.UserRole)  # 获取选中的相似度对

        image1_path = os.path.join(self.image_folder, image1_name)
        image2_path = os.path.join(self.image_folder, image2_name)

        # 显示两张图片
        self.show_image(self.image_label_1, image1_path, side='left')
        self.show_image(self.image_label_2, image2_path, side='right')

        # 显示对应的图片名称
        self.image_name_label_1.setText(image1_name)
        self.image_name_label_2.setText(image2_name)

        # 更新选中的图片
        self.selected_images = (image1_name, image2_name)
        self.selected_image_index = None  # 默认没有选中任何一张

        self.update_image_highlight()

    def update_image_highlight(self):
        """ 更新图片的高亮状态 """
        if self.selected_image_index == 0:
            self.image_label_1.setStyleSheet("border: 5px solid rgba(255, 255, 0, 1);")
            self.image_label_2.setStyleSheet("border: none;")
        elif self.selected_image_index == 1:
            self.image_label_1.setStyleSheet("border: none;")
            self.image_label_2.setStyleSheet("border: 5px solid rgba(255, 255, 0, 1);")
        else:
            self.image_label_1.setStyleSheet("border: none;")
            self.image_label_2.setStyleSheet("border: none;")


        

    def show_image(self, label, image_path, side):
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

        # 为每个图片添加点击事件
        label.mousePressEvent = lambda event, side=side: self.image_clicked(side)

    def image_clicked(self, side):
        """ 处理左侧或右侧图片的点击事件 """
        if side == 'left':
            self.selected_image_index = 0  # 选择左侧图片
        elif side == 'right':
            self.selected_image_index = 1  # 选择右侧图片
        print(f"Selected image index: {self.selected_image_index}")
        self.update_image_highlight()


    def delete_image(self):
        """ 删除选中的图片 """
        if self.selected_image_index is None:
            return

        image_name_to_delete = self.selected_images[self.selected_image_index]
        image_path_to_delete = os.path.join(self.image_folder, image_name_to_delete)

        # 执行删除操作
        if os.path.exists(image_path_to_delete):
            os.remove(image_path_to_delete)
            print(f"Deleted {image_name_to_delete}")

            # 更新相似度对并刷新界面
            self.all_similarities = [
                (sim[0], sim[1], sim[2]) for sim in self.all_similarities if sim[0] != image_name_to_delete and sim[1] != image_name_to_delete
            ]
            self.display_similarities()

            # 清空右侧显示区域
            self.image_label_1.clear()
            self.image_label_2.clear()
            self.image_name_label_1.clear()
            self.image_name_label_2.clear()

    def rename_image(self):
        """ 重命名选中的图片并自动添加文件后缀 """
        if self.selected_image_index is None:
            return

        # 获取选中的图片
        image_name_to_rename = self.selected_images[self.selected_image_index]
        image_path_to_rename = os.path.join(self.image_folder, image_name_to_rename)
        
        # 获取文件的扩展名（后缀）
        file_extension = os.path.splitext(image_name_to_rename)[1]

        # 弹出对话框，让用户输入新名称（不包含扩展名）
        new_name, ok = QInputDialog.getText(self, "Rename Image", "Enter new name:")

        if ok and new_name:
            # 新名称加上扩展名
            new_name_with_extension = new_name + file_extension

            old_path = image_path_to_rename
            new_path = os.path.join(self.image_folder, new_name_with_extension)

            # 避免文件名冲突，检查文件是否存在
            if os.path.exists(new_path):
                QMessageBox.warning(self, "Error", "File already exists!")
            else:
                os.rename(old_path, new_path)
                print(f"Renamed {image_name_to_rename} to {new_name_with_extension}")

                # 更新相似度对并刷新界面
                self.all_similarities = [
                    (sim[0], sim[1], sim[2]) if sim[0] != image_name_to_rename else (new_name_with_extension, sim[1], sim[2])
                    for sim in self.all_similarities
                ]
                self.display_similarities()

                # 清空右侧显示区域
                self.image_label_1.clear()
                self.image_label_2.clear()
                self.image_name_label_1.clear()
                self.image_name_label_2.clear()

    def filter_results(self):
        """ 根据输入的相似度区间过滤结果 """
        min_similarity = self.min_similarity_input.text()
        max_similarity = self.max_similarity_input.text()

        try:
            min_similarity = float(min_similarity) if min_similarity else 0
            max_similarity = float(max_similarity) if max_similarity else 1
        except ValueError:
            return

        filtered_similarities = [
            sim for sim in self.all_similarities
            if min_similarity <= sim[2] <= max_similarity
        ]

        self.result_list.clear()
        for sim in filtered_similarities:
            item = QListWidgetItem(f"{sim[0]} vs {sim[1]}: Similarity = {sim[2]:.4f}")
            item.setData(Qt.UserRole, (sim[0], sim[1], sim[2]))
            self.result_list.addItem(item)

        self.filter_result_label.setText(f"Filtered results: {len(filtered_similarities)} pairs found.")

    def toggle_sort_order(self):
        """ 切换排序顺序（高→低或低→高） """
        self.sort_descending = not self.sort_descending
        self.sort_button.setText(f"Sort Order: {'High → Low' if self.sort_descending else 'Low → High'}")
        self.display_similarities()


if __name__ == '__main__':
    app = QApplication([])
    window = ImageSimilarityTool()
    window.show()
    app.exec_()
