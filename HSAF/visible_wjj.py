import os
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np

from PIL import Image
from torchvision import transforms

from models.resnet import ResNet


# =========================================================
# 1. 构建模型
# =========================================================

model = ResNet(
    depth=18,
    num_classes=512,
    maxpool=False,
    zero_init_residual=True
)

model.fc = nn.Identity()


# =========================================================
# 2. 加载权重
# =========================================================

ckpt_path = "/home/data3t/zhaoyuxiang/CL/ContrastiveCrop-mainn/ContrastiveCrop/checkpoints/small/try/epoch_500.pth"
# /home/data3t/zhaoyuxiang/CL/ContrastiveCrop-mainn/ContrastiveCrop/checkpoints/small/genggai/epoch_300.pth
# "/home/data3t/zhaoyuxiang/CL/ContrastiveCrop-mainn/ContrastiveCrop/checkpoints/small/try/epoch_500.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")

if "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]

elif "simsiam_state" in ckpt:
    state_dict = ckpt["simsiam_state"]

elif "simclr_state" in ckpt:
    state_dict = ckpt["simclr_state"]

elif "moco_state" in ckpt:
    state_dict = ckpt["moco_state"]

elif "byol_state" in ckpt:
    state_dict = ckpt["byol_state"]

else:
    state_dict = ckpt

new_state_dict = {}

for k, v in state_dict.items():

    if k.startswith("module.encoder."):
        k = k.replace("module.encoder.", "")

    if k.startswith("module.encoder_q."):
        k = k.replace("module.encoder_q.", "")

    if k.startswith("module."):
        k = k.replace("module.", "")

    if not k.startswith("fc"):
        new_state_dict[k] = v

msg = model.load_state_dict(
    new_state_dict,
    strict=False
)

print("load result:")
print(msg)

model.eval()


# =========================================================
# 3. hook feature
# =========================================================

features = []

def hook_fn(module, input, output):
    features.clear()
    features.append(output.detach())

hook = model.layer4.register_forward_hook(hook_fn)


# =========================================================
# 4. 数据路径
# =========================================================

input_dir = "/home/data3t/zhaoyuxiang/CL/home/data/STL_images/test/test_"

save_dir = "/home/data3t/zhaoyuxiang/CL/ContrastiveCrop-mainn/ContrastiveCrop/result_fig/baseline"

os.makedirs(save_dir, exist_ok=True)


# =========================================================
# 5. transform
# =========================================================

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4406, 0.4273, 0.3858],
        std=[0.2312, 0.2265, 0.2237]
    )
])


# =========================================================
# 6. 获取所有图片
# =========================================================

img_list = []

img_list += glob.glob(os.path.join(input_dir, "*.jpg"))
img_list += glob.glob(os.path.join(input_dir, "*.png"))
img_list += glob.glob(os.path.join(input_dir, "*.jpeg"))
img_list += glob.glob(os.path.join(input_dir, "*.bmp"))

img_list = sorted(img_list)

print("total images:", len(img_list))


# =========================================================
# 7. 遍历处理
# =========================================================

for idx, img_path in enumerate(img_list):

    print(f"[{idx+1}/{len(img_list)}] processing: {img_path}")

    try:

        # -------------------------------------------------
        # 读取图像
        # -------------------------------------------------

        img = Image.open(img_path).convert("RGB")

        x = transform(img).unsqueeze(0)

        # -------------------------------------------------
        # forward
        # -------------------------------------------------

        with torch.no_grad():
            _ = model(x)

        feat_map = features[0][0]

        # -------------------------------------------------
        # L2 norm heatmap
        # -------------------------------------------------

        heatmap = torch.norm(
            feat_map,
            p=2,
            dim=0
        )

        heatmap -= heatmap.min()

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        heatmap = heatmap.cpu().numpy()

        # -------------------------------------------------
        # resize
        # -------------------------------------------------

        img_np = np.array(img.resize((96, 96)))

        heatmap = cv2.resize(
            heatmap,
            (img_np.shape[1], img_np.shape[0])
        )

        # -------------------------------------------------
        # color map
        # -------------------------------------------------

        heatmap_uint8 = np.uint8(255 * heatmap)

        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        heatmap_color = cv2.cvtColor(
            heatmap_color,
            cv2.COLOR_BGR2RGB
        )

        # -------------------------------------------------
        # overlay
        # -------------------------------------------------

        overlay = cv2.addWeighted(
            img_np,
            0.6,
            heatmap_color,
            0.3,
            0
        )

        # -------------------------------------------------
        # 保存
        # -------------------------------------------------

        save_path = os.path.join(
            save_dir,
            f"{idx:05d}.jpg"
        )

        cv2.imwrite(
            save_path,
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )

    except Exception as e:

        print(f"ERROR: {img_path}")
        print(e)

print("Done.")

hook.remove()