import os
import shutil

val_dir = '/home/data3t/zhaoyuxiang/CL/Data/tiny-imagenet-200/val'
img_dir = os.path.join(val_dir, 'images')
anno_file = os.path.join(val_dir, 'val_annotations.txt')

with open(anno_file, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        img_name = parts[0]
        cls = parts[1]

        cls_dir = os.path.join(val_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)

        src = os.path.join(img_dir, img_name)
        dst = os.path.join(cls_dir, img_name)

        shutil.move(src, dst)