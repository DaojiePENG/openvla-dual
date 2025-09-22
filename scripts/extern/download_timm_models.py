import timm
import torch

# 要下载的模型名称列表
model_names = [
    "vit_tiny_patch16_224",
    "vit_large_patch14_dinov2.lvd142m",
]

for name in model_names:
    # 加载模型（会自动下载权重并缓存）
    model = timm.create_model(
        name,
        pretrained=True,
        num_classes=0
    )
    # 可选：验证模型是否加载成功
    print(f"已下载并缓存模型: {name}")