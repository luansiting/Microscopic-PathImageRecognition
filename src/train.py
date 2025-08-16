import os
import torch
from torch.utils.data import DataLoader
from src.models.model import get_vit_model
from src.utils import (
    PathologyDataset, get_class_names, split_train_val_test,
    calculate_metrics, plot_metrics_curve, save_metrics_results
)

data_root = "D:/Pathology_Diagnosis_Platform/data"
save_model_path = "D:/Pathology_Diagnosis_Platform/deployment/models/trained_model.pth"
results_dir = "D:/Pathology_Diagnosis_Platform/results"
batch_size = 16
epochs = 5
random_seed = None

# 创建结果文件夹
os.makedirs(f"{results_dir}/training", exist_ok=True)
os.makedirs(f"{results_dir}/validation", exist_ok=True)
os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

# 1. 加载并划分数据（80%训练，10%验证，10%测试）
full_dataset = PathologyDataset(data_root)
train_dataset, val_dataset, test_dataset = split_train_val_test(
    full_dataset, seed=random_seed
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
class_names = get_class_names(data_root)
num_classes = len(class_names)

# 2. 初始化模型、优化器、损失函数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_vit_model(num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# 记录指标历史
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

# 3. 训练循环
print(f"使用设备: {device}, 类别数: {num_classes}, 训练样本数: {len(train_dataset)}, "
      f"验证样本数: {len(val_dataset)}, 测试样本数: {len(test_dataset)}")

for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_preds = []
    train_labels = []
    train_total_loss = 0.0

    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device).float(), labels.to(device)

        # 前向传播
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失和预测结果
        train_total_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        train_preds.extend(preds)
        train_labels.extend(labels.cpu().numpy())

        # 每100个批次打印一次进度
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_loader)}, "
                  f"Batch Loss: {loss.item():.4f}")

    # 计算训练集指标
    train_avg_loss = train_total_loss / len(train_loader)
    train_metrics = calculate_metrics(train_preds, train_labels)
    train_loss_history.append(train_avg_loss)
    train_acc_history.append(train_metrics['accuracy'])

    # 验证阶段
    model.eval()
    val_preds = []
    val_labels = []
    val_total_loss = 0.0

    with torch.no_grad():  # 关闭梯度计算，节省内存
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device).float(), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())

    # 计算验证集指标
    val_avg_loss = val_total_loss / len(val_loader)
    val_metrics = calculate_metrics(val_preds, val_labels)
    val_loss_history.append(val_avg_loss)
    val_acc_history.append(val_metrics['accuracy'])

    # 打印本轮完整指标
    print(f"\nEpoch {epoch + 1}/{epochs} 总结:")
    print(f"训练集 - 损失: {train_avg_loss:.4f}, 准确率: {train_metrics['accuracy']:.4f}, "
          f"精确率: {train_metrics['precision']:.4f}, 召回率: {train_metrics['recall']:.4f}, "
          f"F1分数: {train_metrics['f1']:.4f}")
    print(f"验证集 - 损失: {val_avg_loss:.4f}, 准确率: {val_metrics['accuracy']:.4f}, "
          f"精确率: {val_metrics['precision']:.4f}, 召回率: {val_metrics['recall']:.4f}, "
          f"F1分数: {val_metrics['f1']:.4f}\n")

# 4. 保存模型和指标
torch.save(model.state_dict(), save_model_path)
print(f"模型已保存到: {save_model_path}")

# 保存指标曲线
plot_metrics_curve(train_loss_history, "Loss", f"{results_dir}/training/train_loss.png")
plot_metrics_curve(train_acc_history, "Accuracy", f"{results_dir}/training/train_acc.png")
plot_metrics_curve(val_loss_history, "Loss", f"{results_dir}/validation/val_loss.png")
plot_metrics_curve(val_acc_history, "Accuracy", f"{results_dir}/validation/val_acc.png")

# 保存最终指标
save_metrics_results(train_metrics, f"{results_dir}/training/final_train_metrics.txt")
save_metrics_results(val_metrics, f"{results_dir}/validation/final_val_metrics.txt")
