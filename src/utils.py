import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class PathologyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        for label, class_name in enumerate(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(np.array(img)).permute(2, 0, 1) / 255.0
        return img, self.labels[idx]


def get_class_names(root_dir):
    class_names = []
    for class_name in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, class_name)):
            class_names.append(class_name)
    return class_names


def split_train_val_test(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=None):
    """划分训练集(80%)、验证集(10%)、测试集(10%)"""
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    # 先分训练集和剩余部分
    train_dataset, temp_dataset = random_split(
        dataset, [train_size, total_size - train_size], generator=generator
    )
    # 再从剩余部分分验证集和测试集
    val_dataset, test_dataset = random_split(
        temp_dataset, [val_size, test_size], generator=generator
    )
    return train_dataset, val_dataset, test_dataset


def calculate_metrics(preds, labels):
    """计算准确率、精确率、召回率、F1分数"""
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')  # 多类别用加权平均
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def plot_metrics_curve(metrics_history, metric_name, save_path):
    """绘制指标变化曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(metrics_history) + 1), metrics_history)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'Training {metric_name} Curve')
    plt.savefig(save_path)
    plt.close()


def save_metrics_results(metrics, save_path):
    """保存指标结果到文件"""
    with open(save_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
