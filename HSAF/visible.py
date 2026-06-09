import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from models.resnet import ResNet


# =========================================================
# 1. 构建 ResNet18 backbone（去掉 fc）
# =========================================================

model = ResNet(depth=18,num_classes=512, maxpool=False,zero_init_residual=True)

model.fc = nn.Identity()

ckpt_path = "/home/data3t/zhaoyuxiang/CL/ContrastiveCrop-mainn/ContrastiveCrop/checkpoints/small/genggai/epoch_300.pth"
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



features = []

def hook_fn(module, input, output):
    features.append(output.detach())

hook = model.layer4.register_forward_hook(hook_fn)

img_path = "/home/data3t/zhaoyuxiang/CL/home/data/STL_images/test/test_/image_00170.png"  

img = Image.open(img_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((96,96)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4406, 0.4273, 0.3858],
        std=[0.2312, 0.2265, 0.2237]
    )
])

x = transform(img).unsqueeze(0)


with torch.no_grad():
    _ = model(x)

feat_map = features[0][0]

heatmap = torch.norm(
    feat_map,
    p=2,
    dim=0
)

heatmap -= heatmap.min()
heatmap /= heatmap.max()

heatmap = heatmap.cpu().numpy()

img_np = np.array(img.resize((96,96)))

heatmap = cv2.resize(
    heatmap,
    (img_np.shape[1], img_np.shape[0])
)

heatmap_uint8 = np.uint8(255 * heatmap)
heatmap_color = cv2.applyColorMap(
    heatmap_uint8,
    cv2.COLORMAP_JET
)
heatmap_color = cv2.cvtColor(
    heatmap_color,
    cv2.COLOR_BGR2RGB
)
overlay = cv2.addWeighted(
    img_np,
    0.7,
    heatmap_color,
    0.3,
    0
)


plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img_np)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Heatmap")
plt.imshow(heatmap, cmap="jet")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.show()



cv2.imwrite(
    "/home/data3t/zhaoyuxiang/CL/ContrastiveCrop-mainn/ContrastiveCrop/result_pic/genggai_simsiam_4.jpg",
    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
)

print("saved to cam_result.jpg")


hook.remove()