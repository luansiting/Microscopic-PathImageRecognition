import sys
import os
import torch
import torch.nn as nn

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# 导入训练时使用的ViT模型
from src.models.model import vit_b_16, ViT_B_16_Weights

# 配置参数
trained_model_path = os.path.join("models", "trained_model.pth")  # 模型权重路径
num_classes = 5
onnx_path = "model.onnx"  # ONNX模型输出路径
openvino_path = "openvino_model"  # OpenVINO模型输出目录


def load_trained_model():
    # 加载预训练的 ViT - B/16 模型（权重为 ImageNet 预训练）
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    # 获取分类头的输入特征维度
    in_features = model.heads.head.in_features

    # 替换分类头为自定义类别数的全连接层
    model.heads.head = nn.Linear(in_features, num_classes)

    # 加载训练好的权重
    model.load_state_dict(
        torch.load(trained_model_path, map_location=torch.device('cpu')),
        strict=True
    )
    model.eval()
    return model

    # 加载模型权重
    try:
        model.load_state_dict(
            torch.load(trained_model_path, map_location=torch.device('cpu')),
            strict=True  # 严格匹配权重与模型结构
        )
        model.eval()  # 设置为推理模式
        print("✅ 模型权重加载成功")
        return model
    except Exception as e:
        print(f"❌ 模型权重加载失败: {str(e)}")
        print("请检查模型结构是否与训练时完全一致")
        sys.exit(1)


def convert_to_onnx(model):
    """将PyTorch模型转换为ONNX格式，使用最新opset"""
    # 创建输入示例（与训练时的输入尺寸一致）
    dummy_input = torch.randn(1, 3, 224, 224)  # 1张图片，3通道，224x224

    # 转换配置
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=16,  # 最新版本通常支持更多算子
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},  # 支持动态batch size
            "output": {0: "batch_size"}
        }
    )

    if os.path.exists(onnx_path):
        print(f"✅ ONNX模型已保存至: {os.path.abspath(onnx_path)}")
    else:
        print("❌ ONNX模型转换失败")
        sys.exit(1)


def convert_to_openvino():
    """将ONNX模型转换为OpenVINO格式"""
    from openvino.tools import mo
    from openvino.runtime import serialize

    # 转换ONNX模型
    try:
        ov_model = mo.convert_model(
            onnx_path,
            input_shape=[1, 3, 224, 224],  # 明确输入形状
            mean_values=[123.675, 116.28, 103.53],
            scale_values=[58.395, 57.12, 57.375]
        )
    except Exception as e:
        print(f"❌ OpenVINO转换失败: {str(e)}")
        sys.exit(1)

    # 创建保存目录
    os.makedirs(openvino_path, exist_ok=True)

    # 保存OpenVINO模型
    serialize(
        ov_model,
        os.path.join(openvino_path, "model.xml"),
        os.path.join(openvino_path, "model.bin")
    )

    print(f"✅ OpenVINO模型已保存至: {os.path.abspath(openvino_path)}")


if __name__ == "__main__":
    print("===== 开始模型转换 =====")
    model = load_trained_model()
    convert_to_onnx(model)
    convert_to_openvino()
    print("===== 模型转换完成 =====")
