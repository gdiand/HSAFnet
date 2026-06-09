import torch
import os
from PIL import Image
from torchvision import datasets
import numpy as np


class STL10_boxes(datasets.STL10):

    def __init__(
        self,
        split,
        root,
        transform_rcrop,
        **kwargs
    ):
        assert split in ['train', 'test', 'unlabeled', 'train+unlabeled']
        super().__init__(split=split, root=root, **kwargs)

        self.transform_rcrop = transform_rcrop

    def __getitem__(self, index):
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        img_pil = Image.fromarray(np.transpose(img, (1, 2, 0)))
        
        def get_single_view(base_img):
            v_img = self.transform_rcrop(base_img)
            return v_img[0],v_img[1]

        x1,x2 = get_single_view(img_pil)
        # patches1 = self.split_to_nine(x1)
        # patches2 = self.split_to_nine(x2)

        if self.target_transform is not None:
            target = self.target_transform(target)


        return [x1, x2], target
