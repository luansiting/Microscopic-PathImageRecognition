import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from openvino.runtime import Core
from src.utils import (
    PathologyDataset, get_class_names, split_train_val_test,
    calculate_metrics, save_metrics_results
)

# 配置
model_xml = "../deployment/models/model.xml"
model_bin = "../deployment/models/model.bin"
data_root = "../data"  # 与训练时路径一致
results_dir = "../results"
random_seed = None

# 创建推理结果文件夹
os.makedirs(f"{results_dir}/inference", exist_ok=True)

# 1. 加载OpenVINO模型
ie = Core()
model = ie.read_model(model=model_xml, weights=model_bin)
compiled_model = ie.compile_model(model=model, device_name="CPU")
input_layer = next(iter(compiled_model.inputs))
output_layer = next(iter(compiled_model.outputs))

# 2. 获取类别和测试集
class_names = get_class_names(data_root)
full_dataset = PathologyDataset(data_root)
_, _, test_dataset = split_train_val_test(full_dataset, seed=random_seed)
test_loader = DataLoader(test_dataset, batch_size=1)

# 3. 在测试集上评估
test_preds = []
test_labels = []
test_results = []  # 保存详细结果

for idx, (imgs, labels) in enumerate(test_loader):
    # 图像预处理
    img_np = imgs[0].numpy().transpose(1, 2, 0)  # (3,224,224) → (224,224,3)
    img_np = cv2.resize(img_np, (224, 224))
    img_np = img_np.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0

    # 推理
    result = compiled_model([img_np])[output_layer]
    predicted = result[0].argmax()
    confidence = result[0].max()

    # 记录结果
    true_label = labels[0].item()
    test_preds.append(predicted)
    test_labels.append(true_label)
    test_results.append({
        "image_path": test_dataset.dataset.images[test_dataset.indices[idx]],
        "true_label": class_names[true_label],
        "predicted_label": class_names[predicted],
        "confidence": float(confidence)
    })

# 计算测试集指标
test_metrics = calculate_metrics(test_preds, test_labels)
print("\n测试集最终指标:")
print(f"准确率: {test_metrics['accuracy']:.4f}, 精确率: {test_metrics['precision']:.4f}, "
      f"召回率: {test_metrics['recall']:.4f}, F1分数: {test_metrics['f1']:.4f}")

# 保存结果
save_metrics_results(test_metrics, f"{results_dir}/inference/test_metrics.txt")
pd.DataFrame(test_results).to_csv(f"{results_dir}/inference/prediction_details.csv", index=False)
print(f"测试结果已保存到: {results_dir}/inference/")


# 4. 单张图像推理函数
def predict(image_path):
    if not os.path.exists(image_path):
        return "图像路径不存在"
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0
    result = compiled_model([img])[output_layer]
    predicted_class = class_names[result[0].argmax()]
    confidence = result[0].max()
    return f"预测类别: {predicted_class}, 置信度: {confidence:.4f}"

