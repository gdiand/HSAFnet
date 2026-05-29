from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
from torchvision.datasets import ImageFolder

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

from models.resnet import ResNet

device = "cuda"

# ======================
# load weights
# ======================

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

# ======================
# model
# ======================

# model_path = '/home/data3t/zhaoyuxiang/CL/ckpt/baseline_tinyIN/epoch_500.pth'
# model_path='/home/data3t/zhaoyuxiang/CL/ckpt/genggai_tinyIN/epoch_300.pth'
# model_path='/home/data3t/zhaoyuxiang/CL/ckpt/baseline_moco/tiny/epoch_500.pth'
model_path= "/home/data3t/zhaoyuxiang/CL/ckpt/genggai_moco/tiny_yuzhi/5-9.5/epoch_300.pth"

model = ResNet(
    depth=18,
    num_classes=200,
    maxpool=False,
    zero_init_residual=True
).to(device)

load_weights(model_path, model)

model.fc = torch.nn.Identity()

model.eval()

# ======================
# dataset
# ======================

mean = (0.4802, 0.4481, 0.3975)
std = (0.2302, 0.2265, 0.2262)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

train_set = ImageFolder(root='/home/data3t/zhaoyuxiang/CL/Data/tiny-imagenet-200/train',transform=transform)

test_set = ImageFolder(root='/home/data3t/zhaoyuxiang/CL/Data/tiny-imagenet-200/val',transform=transform)

train_loader = DataLoader(
    train_set,
    batch_size=512,
    shuffle=False,
    num_workers=8
)

test_loader = DataLoader(
    test_set,
    batch_size=512,
    shuffle=False,
    num_workers=8
)

# ======================
# extract train feature
# ======================

train_features = []
train_labels = []

with torch.no_grad():

    for image, label in train_loader:

        image = image.to(device)

        feat = model(image)

        feat = F.normalize(feat, dim=1)

        train_features.append(feat.cpu())

        train_labels.append(label)

train_features = torch.cat(train_features, dim=0)
train_labels = torch.cat(train_labels, dim=0)

print("train feature shape:", train_features.shape)

# ======================
# knn evaluation
# ======================

k = 200
total = 0
correct = 0

with torch.no_grad():

    for image, label in test_loader:

        image = image.to(device)

        feat = model(image)

        feat = F.normalize(feat, dim=1)

        # cosine similarity
        sim_matrix = torch.mm(feat, train_features.T.to(device))

        # top-k nearest
        sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)

        sim_labels = train_labels[sim_indices.cpu()]

        # vote
        pred_labels = []

        for i in range(sim_labels.shape[0]):

            counts = torch.bincount(sim_labels[i], minlength=10)

            pred = counts.argmax()

            pred_labels.append(pred)

        pred_labels = torch.tensor(pred_labels)

        correct += (pred_labels == label).sum().item()

        total += label.size(0)

acc = correct / total * 100

print(f"KNN Accuracy: {acc:.2f}%")