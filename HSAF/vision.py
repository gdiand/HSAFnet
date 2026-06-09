import torch
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# -----------------------------
# 1. 加载你的模型（直接用你写的ResNet类）
# -----------------------------
from models import ResNet   # ← 改成你的路径


# -----------------------------
# 2. 加载权重
# -----------------------------
def load_weights(model, path):
    state_dict = torch.load(path, map_location="cpu")

    new_dict = {}
    for k, v in state_dict.items():

        # 去掉 encoder / module 前缀
        if k.startswith("encoder."):
            k = k.replace("encoder.", "")
        if k.startswith("module."):
            k = k.replace("module.", "")

        # 忽略 projector / predictor
        if any(x in k for x in ["projector", "predictor"]):
            continue

        new_dict[k] = v

    model.load_state_dict(new_dict, strict=False)
    print("权重加载完成")


# -----------------------------
# 3. 可视化
# -----------------------------
def visualize(act, path="/home/data3t/zhaoyuxiang/CL/ContrastiveCrop-mainn/ContrastiveCrop/pic"):
    save_dir = path
    os.makedirs(save_dir, exist_ok=True)

    # -----------------------
    # 1. 维度处理
    # -----------------------
    if len(act.shape) == 3:
        act = np.mean(act, axis=0)  # C,H,W → H,W

    # -----------------------
    # 2. ReLU
    # -----------------------
    act = np.maximum(act, 0)


    p_min, p_max = np.percentile(act, (5, 95))
    act = np.clip((act - p_min) / (p_max - p_min + 1e-8), 0, 1)

    # 如果是 4维 (1,C,H,W)
    if len(act.shape) == 4:
        act = act[0]   # → (C,H,W)

    # 如果是 3维 (C,H,W)
    if len(act.shape) == 3:
        act = np.mean(act, axis=0)  # → (H,W)

    # -----------------------
    # 4. 可视化
    # -----------------------
    plt.imshow(act, cmap='jet')
    plt.colorbar()
    plt.title("Activation Map")

    # -----------------------
    # 5. 保存
    # -----------------------
    num_files = len(os.listdir(save_dir))
    save_path = os.path.join(save_dir, f"{num_files}.png")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"结果已保存至: {save_path}")


#/home/data3t/zhaoyuxiang/CL/ContrastiveCrop-mainn/ContrastiveCrop/checkpoints/small/try/epoch_500.pth
#/home/data3t/zhaoyuxiang/CL/ContrastiveCrop-mainn/ContrastiveCrop/checkpoints/small/try_200epoch/epoch_300.pth
# -----------------------------
# 4. 主函数
# -----------------------------
def main():
    img_path = "/home/data3t/zhaoyuxiang/CL/home/data/STL_images/train/train_/image_00001.png"        # ← 改
    weight_path = "/home/data3t/zhaoyuxiang/CL/ContrastiveCrop-mainn/ContrastiveCrop/checkpoints/small/try/epoch_500.pth"    # ← 改

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(
        depth=18,
        num_classes=128,   # dim（随便填，不影响）
        maxpool=False,
        zero_init_residual=True
    ).to(device)

    load_weights(model, weight_path)
    model.eval()

    # 预处理
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=96, scale=(0.2, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.4406, 0.4273, 0.3858),std = (0.2312, 0.2265, 0.2237))])

    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(x, return_feat=True)

    visualize(feat.detach().cpu().numpy())


if __name__ == "__main__":
    main()