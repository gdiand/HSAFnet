import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
from torchvision import datasets, transforms
from PIL import Image
from resnet import ResNet 
from tqdm import tqdm


def get_model(path):
    # 使用你提供的 ResNet 初始化参数
    model = ResNet(
        depth=18,
        num_classes=128,   
        maxpool=False,
        zero_init_residual=True
    )
    model.fc = nn.Identity()
    
    checkpoint = torch.load(path, map_location="cpu")
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
    return model.cuda()

# ================= 2. 核心量化函数 =================
def compute_iou(model, img, gt_mask):

    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4406, 0.4273, 0.3858), std=(0.2312, 0.2265, 0.2237))
    ])
    
    input_tensor = transform(img).unsqueeze(0).cuda()

    with torch.no_grad():

        x = model.conv1(input_tensor)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        features = model.layer4(x).squeeze(0) 

    c, h, w = features.shape
    feat_flat = features.view(c, -1).permute(1, 0).cpu().numpy()
    

    gt_array = np.array(gt_mask)
    gt_h, gt_w = gt_array.shape

    binary_gt = ((gt_array > 0) & (gt_array != 255)).astype('uint8')
    
    best_iou = 0.0
    

    for k in [2, 3, 4]:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=5)
        labels = kmeans.fit_predict(feat_flat)
        mask_all = labels.reshape(h, w)


        for cluster_id in range(k):

            current_cluster_mask = (mask_all == cluster_id).astype('uint8')
            

            mask_rescaled = cv2.resize(current_cluster_mask, (gt_w, gt_h), interpolation=cv2.INTER_NEAREST)
            

            intersection = np.logical_and(mask_rescaled, binary_gt).sum()
            union = np.logical_or(mask_rescaled, binary_gt).sum()
            iou = intersection / (union + 1e-6)
            

            if iou > best_iou:
                best_iou = iou
                
    return best_iou

# ================= 3. 批量评估 =================
if __name__ == "__main__":

    CKPT_LIST = [
        "/home/data3t/zhaoyuxiang/CL/ContrastiveCrop-mainn/ContrastiveCrop/checkpoints/small/try/epoch_500.pth",
        "/home/data3t/zhaoyuxiang/CL/ContrastiveCrop-mainn/ContrastiveCrop/checkpoints/small/genggai/epoch_300.pth"
    ]
    VOC_ROOT = "/home/data3t/zhaoyuxiang/CL"
    

    dataset = datasets.VOCSegmentation(VOC_ROOT, year='2012', image_set='val', download=False)
    
    for ckpt in CKPT_LIST:
        print(f"\n正在评估权重: {os.path.basename(ckpt)}")
        model = get_model(ckpt)
        
        ious = []

        num_test = 1449
        
        for i in tqdm(range(num_test)):
            img, target = dataset[i]

            if np.max(np.array(target)) == 0:
                continue
            
            res_iou = compute_iou(model, img, target)
            ious.append(res_iou)
        
        print(f"权重 {os.path.basename(ckpt)} 的平均物体发现 IoU: {np.mean(ious):.4f}")