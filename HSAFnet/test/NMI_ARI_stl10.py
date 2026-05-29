from sklearn.cluster import KMeans
from torch_clustering import PyTorchKMeans,evaluate_clustering

from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

from models.resnet import ResNet
from models.simsiam import SimSiam
from torchvision import datasets

device="cuda"

def load_weights(ckpt_path, model):
    # load checkpoint
    print("==> Loading checkpoint '{}'".format(ckpt_path))
    assert os.path.isfile(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cuda')


    if 'simclr_state' in ckpt.keys():  # simclr
            state_dict = ckpt['simclr_state']
            new_state_dict = {}
            for k, v in state_dict.items():
                newk = k
                if 'fc.' in newk:
                    continue
                new_state_dict[newk] = v
            del state_dict
    elif 'simsiam_state' in ckpt.keys():  # simsiam
            state_dict = ckpt['simsiam_state']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.encoder.') and not k.startswith('module.encoder.fc'):
                    newk = k.replace('module.encoder.', '')
                    new_state_dict[newk] = v
            del state_dict
    else:  # moco & byol
            for k in ['moco_state', 'byol_state']:
                if k in ckpt.keys():
                    state_dict = ckpt[k]
                    break
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.encoder_q.') and not k.startswith('module.encoder_q.fc'):
                    newk = k.replace('module.encoder_q.', '')
                    new_state_dict[newk] = v
            del state_dict

    msg = model.load_state_dict(new_state_dict, strict=False)
    assert set(msg.missing_keys) == {'fc.weight', 'fc.bias'}, set(msg.missing_keys)

    start_epoch = ckpt['epoch'] + 1
    print("Model weights loaded. (epoch {})".format(ckpt['epoch']))
    return start_epoch


# model_path = '/home/data3t/zhaoyuxiang/CL/ContrastiveCrop-mainn/ContrastiveCrop/checkpoints/small/try/epoch_500.pth'
# model_path = "/home/data3t/zhaoyuxiang/CL/ContrastiveCrop-mainn/ContrastiveCrop/checkpoints/small/genggai/epoch_300.pth"

model_path='/home/data3t/zhaoyuxiang/CL/ckpt/genggai_moco/epoch_300.pth'
model=ResNet(depth=18,num_classes=10,maxpool=False,zero_init_residual=True).to(device)


load_weights(model_path, model)

model.eval()
model.fc = torch.nn.Identity()

mean = (0.4406, 0.4273, 0.3858)
std = (0.2312, 0.2265, 0.2237)

transfomer=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
data=datasets.STL10(split='test',root='/home/data3t/zhaoyuxiang/CL/simsiam/SimSiam-main/data/STL_images_bin',transform=transfomer)

dataloader=DataLoader(data,batch_size=512,shuffle=False,num_workers=8)


features=[]
labels=[]

for index,(image,label) in enumerate(dataloader):
    image=image.to(device)
    img=model(image)
    img = img.detach().cpu().numpy()

    features.append(img)
    labels.append(label.numpy())

features = np.concatenate(features, axis=0)
labels = np.concatenate(labels, axis=0)

features=normalize(features)
# features = torch.tensor(features).float().cuda()
num_classes = len(np.unique(labels))
print("num_classes:", num_classes)


kmeans = KMeans(
    n_clusters=num_classes,
    random_state=0,
    n_init=10
)

pred = kmeans.fit_predict(features)

# clustering_model = PyTorchKMeans(init='k-means++',n_clusters=num_classes,n_init=10)
# pred= clustering_model.fit_predict(features)
# pred = pred.cpu().numpy()

nmi = normalized_mutual_info_score(labels, pred)

ari = adjusted_rand_score(labels, pred)


print(f"NMI: {nmi}")
print(f"ARI: {ari}")