from __future__ import print_function

import os
import os.path
import numpy as np
import random
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

import distillation.utils as utils

# Set the appropriate paths of the datasets here.
_IMAGENET_DATASET_DIR = 'datasets/tiny-imagenet-200'
# _MEAN_PIXEL = [0.485, 0.456, 0.406]
# _STD_PIXEL = [0.229, 0.224, 0.225]
_MEAN_PIXEL = [0.4802, 0.4481, 0.3975]
_STD_PIXEL = [0.2302, 0.2265, 0.2262]


class TinyImageNetBase(data.Dataset):
    def __init__(
        self,
        data_dir=_IMAGENET_DATASET_DIR,
        split='train',
        transform=None):
        # assert (split in ('train', 'val', 'test')) 
        self.split = split
        self.name = f'TinyImageNet_Split_' + self.split

        print(f'==> Loading TinyImageNet dataset - split {self.split}')
        print(f'==> ImageNet directory: {data_dir}')

        self.transform = transform
        print(f'==> transform: {self.transform}')
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        test_dir = os.path.join(data_dir, 'test')
        
        if (split == 'train') or (split == 'train_subset'):        
            self.data = datasets.ImageFolder(train_dir, self.transform)
        elif split == 'val':        
            self.data = datasets.ImageFolder(val_dir, self.transform)
        else:
            self.data = datasets.ImageFolder(test_dir, self.transform)
            
        self.labels = [item[1] for item in self.data.imgs]
        
        if (split == 'train_subset'):
            subsetK = 200
            assert subsetK > 0

            label2ind = utils.buildLabelIndex(self.data.targets)
            all_indices = []
            for label, img_indices in label2ind.items():
                assert len(img_indices) >= subsetK
                all_indices += img_indices[:subsetK]

            self.data.imgs = [self.data.imgs[idx] for idx in  all_indices]
            self.data.samples = [self.data.samples[idx] for idx in  all_indices]
            self.data.targets = [self.data.targets[idx] for idx in  all_indices]
            self.labels = [self.labels[idx] for idx in  all_indices]
        

    def __getitem__(self, index):
        img, label = self.data[index]
        return img, label

    def __len__(self):
        return len(self.data)


class TinyImageNet(TinyImageNetBase):
    def __init__(
        self,
        data_dir=_IMAGENET_DATASET_DIR,
        split='train',
        do_not_use_random_transf=False):
        transform_train = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL),
        ])
        if do_not_use_random_transf or (split == 'val') or (split == 'test'):
            transform = transform_test
        else:
            transform = transform_train
        TinyImageNetBase.__init__(self, data_dir=data_dir, split=split, transform=transform)

# class TinyImageNet(TinyImageNetBase):
#     def __init__(
#         self,
#         data_dir=_IMAGENET_DATASET_DIR,
#         split='train',
#         do_not_use_random_transf=False):
#         transform_train = transforms.Compose([
#             transforms.RandomResizedCrop(64),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor()#,
#             #transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL),
#         ])
#         transform_test = transforms.Compose([
#             transforms.ToTensor()#,
#             #transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL),
#         ])
#         if do_not_use_random_transf or (split == 'val') or (split == 'test'):
#             transform = transform_test
#         else:
#             transform = transform_train
#         TinyImageNetBase.__init__(self, data_dir=data_dir, split=split, transform=transform)

