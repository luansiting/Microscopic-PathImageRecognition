import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights


def get_vit_model(num_classes):
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    # 替换最后一层以适应自定义类别数
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    return model
