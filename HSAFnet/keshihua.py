import matplotlib.pyplot as plt
import os
import torch

def vision(act,path):
    save_dir = path
    os.makedirs(save_dir, exist_ok=True)

    plt.imshow(act, cmap='jet')
    plt.colorbar()
    plt.title("Activation Map")

    num_files = len(os.listdir(save_dir))
    save_path = os.path.join(save_dir, f"{num_files}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close() 

def keshihua(feature_map):
    C, H, W=feature_map.shape
    xianzhu=torch.norm(feature_map, p=2, dim=0)
    pic = xianzhu.numpy()
    vision(pic,"/home/data3t/zhaoyuxiang/CL/ContrastiveCrop-mainn/ContrastiveCrop/pic/1.jpg")


