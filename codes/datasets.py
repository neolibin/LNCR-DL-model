# -*- coding: UTF-8 -*-

import os
import random
from torch.utils.data import Dataset, ConcatDataset
import torchvision.datasets as datasets
import torchvision
import numpy as np
from PIL import Image


import cv2
import pandas as pd
import random


def RGBcompute(path): 
    image_Rmean = []
    image_Gmean = []
    image_Bmean = []
    image_Rstd = []
    image_Gstd = []
    image_Bstd = []
    N=0
    
    print('Compute RGB mean and std')
    for image in os.listdir(os.path.join(path, 'patches')): 
        if image.split('.')[-1]=='jpeg':
            img = cv2.imread(os.path.join(path, 'patches', image), 1)
            image_Rmean.append(np.mean(img[:,:,0]))
            image_Gmean.append(np.mean(img[:,:,1]))
            image_Bmean.append(np.mean(img[:,:,2]))
            image_Rstd.append(np.std(img[:,:,0]))
            image_Gstd.append(np.std(img[:,:,1]))
            image_Bstd.append(np.std(img[:,:,2]))
        else:
            pass

    R_mean = np.mean(image_Rmean)/255
    G_mean = np.mean(image_Gmean)/255
    B_mean = np.mean(image_Bmean)/255
    R_std = np.mean(image_Rstd)/255
    G_std = np.mean(image_Gstd)/255
    B_std = np.mean(image_Bstd)/255
    print(f'RGB mean={R_mean}, {G_mean}, {B_mean}; RGB std=, {R_std}, {G_std}, {B_std}')
    
    return R_mean, G_mean, B_mean, R_std, G_std, B_std

def channel_shuffle_fn(img):
    img = np.array(img, dtype=np.uint8)

    channel_idx = list(range(img.shape[-1]))
    random.shuffle(channel_idx)

    img = img[:, :, channel_idx]

    img = Image.fromarray(img, 'RGB')
    return img


class ClusterDataset(Dataset):
    def __init__(self, root, dataset_type, training=True):
        self.training = training
        self.dataset = datasets.ImageFolder(root)
        self.R_mean, self.G_mean, self.B_mean, self.R_std, self.G_std, self.B_std = RGBcompute(root)
        
        if dataset_type.split('_')[0]=='PAS':
            bright = 0.1
        else:
            bright = 0.4
            
        print(bright)
        
        img_size = 224
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(384),
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[self.R_mean, self.G_mean, self.B_mean], std=[self.R_std, self.G_std, self.B_std]),
        ])

        self.transforms_aug = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(384),
            torchvision.transforms.Resize(224),
            torchvision.transforms.RandomResizedCrop(img_size, scale=(0.4, 1.0), ratio=(3. / 4., 4. / 3.)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            channel_shuffle_fn,
            torchvision.transforms.ColorJitter(brightness=bright, contrast=0.4, saturation=0.4, hue=0.125),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[self.R_mean, self.G_mean, self.B_mean], std=[self.R_std, self.G_std, self.B_std]),
        ])
        
        
#         imgs = []
#         for line in os.listdir(root):
#             imgs.append(root+line)
#         self.imgs = imgs
            

#     def __getitem__(self, item):
#         fn = self.imgs[item]
#         img_raw = Image.open(fn).convert('RGB')
#         img = self.transforms(img_raw)
#         if self.training:
#             img_aug = self.transforms_aug(img_raw)
#             return img, img_aug
#         else:
#             return img, fn
        
        
#     def __len__(self):
#         return len(self.imgs)


    def __getitem__(self, item):
        img_raw, label = self.dataset[item]
        img = self.transforms(img_raw)
        if self.training:
            img_aug = self.transforms_aug(img_raw)
            return img, img_aug
        else:
            return img, label

    def __len__(self):
        return len(self.dataset)
