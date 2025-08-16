import cv2
import numpy as np
from openvino.runtime import Core

# 1. 加载 OpenVINO 模型
core = Core()
model_xml = "deployment/openvino_model/model.xml"  # 模型路径
model = core.read_model(model=model_xml)
compiled_model = core.compile_model(model=model, device_name="CPU")  # 可指定设备

# 获取输入和输出节点
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)


# 2. 图像预处理函数
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB
    image = cv2.resize(image, (224, 224))  # 调整尺寸
    image = image / 255.0  # 归一化
    mean = [0.485, 0.456, 0.406]  # 与训练时的均值一致
    std = [0.229, 0.224, 0.225]  # 与训练时的标准差一致
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))  # 调整通道顺序为 (C, H, W)
    image = np.expand_dims(image, axis=0)  # 添加 batch 维度
    return image.astype(np.float32)


# 3. 推理函数
def infer(image_path):
    input_image = preprocess_image(image_path)
    result = compiled_model([input_image])[output_layer]
    predicted_class = np.argmax(result)
    return predicted_class


# 4. 测试推理
if __name__ == "__main__":
    # 测试图像路径
    test_image_path = "D:/Pathology_Diagnosis_Platform/data/colon_n/colonn36.jpeg"

    # 类别名称映射
    class_names = ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]

    try:
        predicted_class_idx = infer(test_image_path)
        # 输出预测结果
        print(f"预测类别索引: {predicted_class_idx}")
        print(f"预测类别: {class_names[predicted_class_idx]}")
    except Exception as e:
        print(f"推理失败: {e}")
