import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from torchvision import models, transforms
from PIL import Image
from models import ResNet

# ================= 1. 修正后的加载逻辑 =================
def get_model(path):
    model = ResNet(
        depth=18,
        num_classes=128,   # dim（随便填，不影响）
        maxpool=False,
        zero_init_residual=True
    )
    model.fc = nn.Identity()
    
    checkpoint = torch.load(path, map_location="cpu")
    # 核心：根据你的报错，进入 simsiam_state
    state_dict = checkpoint.get('simsiam_state', checkpoint.get('state_dict', checkpoint))

    new_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("encoder.", "").replace("backbone.", "")
        if any(x in k for x in ["projector", "predictor", "fc"]):
            continue
        new_dict[k] = v

    msg = model.load_state_dict(new_dict, strict=False)
    print(f"权重加载结果: {msg}")
    model.eval()
    return model

# ================= 2. 保存单张结果的函数 =================
def save_kmeans_res(model, img_path, save_path):
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4406, 0.4273, 0.3858), std=(0.2312, 0.2265, 0.2237))
    ])
    
    raw_img = Image.open(img_path).convert('RGB')
    input_tensor = transform(raw_img).unsqueeze(0)

    with torch.no_grad():
        # 手动前向传播到 layer4
        x = model.conv1(input_tensor)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        features = model.layer4(x) 

    # K-Means 聚类
    c, h, w = features.shape[1:]
    feat_flat = features.view(c, -1).permute(1, 0).cpu().numpy()
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(feat_flat)
    mask = labels.reshape(h, w)
    
    # 简单的自动翻转逻辑：假设角落（0,0）大概率是背景
    if mask[0, 0] == 1: 
        mask = 1 - mask

    mask_up = cv2.resize(mask.astype('uint8'), (96, 96), interpolation=cv2.INTER_NEAREST)

    # 绘图并保存
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(raw_img.resize((96, 96)))
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(raw_img.resize((96, 96)))
    axes[1].imshow(mask_up, alpha=0.5, cmap='jet')
    axes[1].set_title("Segmentation Mask")
    axes[1].axis('off')

    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close() # 记得关闭，否则内存会爆

# ================= 3. 主程序 =================
# /home/data3t/zhaoyuxiang/CL/ContrastiveCrop-mainn/ContrastiveCrop/checkpoints/small/genggai/epoch_300.pth
#/home/data3t/zhaoyuxiang/CL/ContrastiveCrop-mainn/ContrastiveCrop/checkpoints/small/try/epoch_500.pth
if __name__ == "__main__":
    MY_CKPT = "/home/data3t/zhaoyuxiang/CL/ContrastiveCrop-mainn/ContrastiveCrop/checkpoints/small/try/epoch_500.pth"
    # 存放结果的文件夹
    out_dir = "/home/data3t/zhaoyuxiang/CL/ContrastiveCrop-mainn/ContrastiveCrop/pic/k_means_baseline"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 如果你有 STL-10 的图片文件夹，可以遍历它
    # 这里演示手动指定几张
    test_images = ["/home/data3t/zhaoyuxiang/CL/ContrastiveCrop-mainn/ContrastiveCrop/pic/orig/1.jpg"] 
    
    m = get_model(MY_CKPT)
    
    for i, img_p in enumerate(test_images):
        save_name = os.path.join(out_dir, f"res_{i}.png")
        save_kmeans_res(m, img_p, save_name)
        print(f"已保存: {save_name}")