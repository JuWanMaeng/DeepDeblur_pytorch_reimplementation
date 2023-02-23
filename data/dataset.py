import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms,models
import json
from data import common
from PIL import Image as im
import cv2
import numpy as np
class train_dataset(Dataset):
    """GOPRO_Large train, test subset class
    """
    def __init__(self):
        super(train_dataset, self).__init__()
        with open('json/train_blur.json', 'r') as f:
            self.train_blur=json.load(f)
        with open('json/train_sharp.json','r') as f:
            self.train_sharp=json.load(f)
        self.transform=transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.train_blur)

    

    def __getitem__(self, idx):
        blur_path, sharp_path=self.train_blur[idx]['img'],self.train_sharp[idx]['img']
        blur,sharp=cv2.imread(blur_path),cv2.imread(sharp_path)
        blur,sharp=cv2.cvtColor(blur,cv2.COLOR_BGR2RGB),cv2.cvtColor(sharp,cv2.COLOR_BGR2RGB)
        
        imgs=[blur,sharp]
        
        imgs=common.crop(*imgs,ps=256)
        
        imgs = common.augment(*imgs, hflip=True, rot=True, shuffle=True, change_saturation=True, rgb_range=255)
        
        imgs[0]=common.add_noise(imgs[0],sigma_sigma=2,rgb_range=255)
        
        imgs=common.generate_pyramid(*imgs,n_scales=3)
        imgs=common.np2tensor(*imgs)
        blur,sharp=imgs[0],imgs[1]
        

        return blur, sharp


class val_dataset(Dataset):
    """GOPRO_Large train, test subset class
    """
    def __init__(self):
        super(val_dataset, self).__init__()
        with open('json/val_blur.json','r') as f:
            self.val_blur=json.load(f)
        with open('json/val_sharp.json','r') as f:
            self.val_sharp=json.load(f)


    def __len__(self):
        return len(self.val_blur)

    def __getitem__(self, idx):
        blur_path, sharp_path=self.val_blur[idx]['img'],self.val_sharp[idx]['img']
        blur,sharp=cv2.imread(blur_path),cv2.imread(sharp_path)
        blur,sharp=cv2.cvtColor(blur,cv2.COLOR_BGR2RGB),cv2.cvtColor(sharp,cv2.COLOR_BGR2RGB)
        imgs=[blur,sharp]
        
        imgs=common.generate_pyramid(*imgs,n_scales=3)
        imgs=common.np2tensor(*imgs)
        blur,sharp=imgs[0],imgs[1]

        return blur, sharp


class test_dataset(Dataset):
    """GOPRO_Large train, test subset class
    """
    def __init__(self):
        super(test_dataset, self).__init__()
        with open('json/test_blur.json','r') as f:
            self.test_blur=json.load(f)
        with open('json/test_sharp.json','r') as f:
            self.test_sharp=json.load(f)


    def __len__(self):
        return len(self.test_blur)

    def __getitem__(self, idx):
        blur_path, sharp_path=self.test_blur[idx]['img'],self.test_sharp[idx]['img']
        blur,sharp=cv2.imread(blur_path),cv2.imread(sharp_path)
        blur,sharp=cv2.cvtColor(blur,cv2.COLOR_BGR2RGB),cv2.cvtColor(sharp,cv2.COLOR_BGR2RGB)
        imgs=[blur,sharp]
        
        # # change saturation
        # amp_factor=np.random.uniform(0.5,1.5)
        # hsv_img=rgb2hsv(img)
        # hsv_img[...,1] *= amp_factor
        # img=

        imgs=common.generate_pyramid(*imgs,n_scales=3)
        imgs=common.np2tensor(*imgs)
        blur,sharp=imgs[0],imgs[1]

        return blur, sharp
