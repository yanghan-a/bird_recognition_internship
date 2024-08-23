import json
import os
import sys
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from torchvision import transforms
# 获取 main.py 所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(current_dir)  # 获取 project 目录的路径
# 将 project 目录添加到 sys.path 中
sys.path.append(main_dir)
from model_v3 import mobilenet_v3_large,mobilenet_v3_small

# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载模型
# model = mobilenet_v3_large(num_classes=2).to(device)
model = mobilenet_v3_small(num_classes=2).to(device)


# model_weight_path = "./MobileNetV3_Small_distortion.pth"
model_weight_path = "./MobileNetV3_Small_5.5_4.4_distortion.pth"






model.load_state_dict(torch.load(model_weight_path))

# 3. 定义数据加载器
data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))  # 获取类别列表
        self.images = []
        self.labels = []

        # 遍历每个类别文件夹
        for class_idx, class_name in enumerate(self.classes):
            class_folder = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                self.images.append(img_path)
                self.labels.append(class_idx)  # 用整数表示类别

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# data_dir = '../data_set/real_data/real_photos/'
data_dir = '../data_test/new_data_test_distortion/'

custom_dataset = CustomDataset(root_dir=data_dir, transform=data_transform)
custom_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

# 4. 提取特征和标签
def extract_features_and_labels(model, data_loader, device):
    model.eval()
    features = []
    labels = []
    predicted_labels = []
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())
            predicted_labels.append(torch.argmax(outputs, dim=1).cpu().numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    predicted_labels = np.concatenate(predicted_labels, axis=0)
    
    return features, labels, predicted_labels

features, labels, predicted_labels = extract_features_and_labels(model, custom_loader, device)

# 5. 应用 t-SNE 降维
def apply_tsne(features):
    tsne = TSNE(n_components=2, perplexity=50, random_state=42)
    features_2d = tsne.fit_transform(features)
    return features_2d

features_2d = apply_tsne(features)

# 6. 可视化 t-SNE 结果
def plot_tsne(features_2d, labels):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    colors = ['blue', 'red']

    class_id_0 = 0
    indices_0 = np.where(labels == class_id_0)
    axes[0].scatter(features_2d[indices_0, 0], features_2d[indices_0, 1], color=colors[0], label=f'Class {class_id_0}', s=50)
    axes[0].legend()
    axes[0].set_title(f'{model_weight_path}')
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')

    class_id_1 = 1
    indices_1 = np.where(labels == class_id_1)
    axes[1].scatter(features_2d[indices_1, 0], features_2d[indices_1, 1], color=colors[1], label=f'Class {class_id_1}', s=50)
    axes[1].legend()
    axes[1].set_title(f'{model_weight_path}')
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 2')

    x_min, x_max = np.min(features_2d[:, 0]), np.max(features_2d[:, 0])
    y_min, y_max = np.min(features_2d[:, 1]), np.max(features_2d[:, 1])

    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    axes[2].scatter(features_2d[indices_0, 0], features_2d[indices_0, 1], color=colors[0], label=f'Class {class_id_0}', s=50, alpha=0.6)
    axes[2].scatter(features_2d[indices_1, 0], features_2d[indices_1, 1], color=colors[1], label=f'Class {class_id_1}', s=50, alpha=0.6)
    axes[2].legend()
    axes[2].set_title(f'{model_weight_path}')
    axes[2].set_xlabel('Dimension 1')
    axes[2].set_ylabel('Dimension 2')

    plt.show()

plot_tsne(features_2d, labels)

# 7. 计算并绘制混淆矩阵及正确/错误率
def plot_confusion_matrix_and_accuracy(labels, predicted_labels):
    cm = confusion_matrix(labels, predicted_labels)
    accuracy = cm.diagonal() / cm.sum(axis=1)
    error_rate = 1 - accuracy

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)

    for i in range(len(accuracy)):
        plt.text(i, i, f'\nAccuracy: {accuracy[i]:.2f}\nError: {error_rate[i]:.2f}', 
                 ha='center', va='center', color='black', fontsize=10, weight='bold')

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix with Accuracy/Error for {model_weight_path}')
    plt.show()

plot_confusion_matrix_and_accuracy(labels, predicted_labels)