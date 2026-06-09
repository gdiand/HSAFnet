import torch
from torchvision import datasets


class Tiny200_boxes(datasets.ImageFolder):
    def __init__(self, root, transform_rcrop, **kwargs):
        super().__init__(root=root, **kwargs)
        self.transform_rcrop = transform_rcrop

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)


        img = self.transform_rcrop(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
