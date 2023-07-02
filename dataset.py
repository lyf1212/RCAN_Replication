import torch
import torchvision.transforms as tfs
import os
import cv2
import numpy as np
from PIL import Image

class DIV2K(torch.utils.data.Dataset):
    def __init__(self, scale, train=True):
        super().__init__()
        self.scale = scale
        self.train = train
    
        self.img_list = []
        
        if self.train:
            self.img_list = os.listdir('./DIV2K_train_LR_bicubic/X{}_sub/'.format(self.scale))
        else:
            self.img_list = os.listdir('./DIV2K_valid_LR_bicubic/X{}_sub/'.format(self.scale))

    def __getitem__(self, index):
        if self.train:
            lr_img = cv2.imread(os.path.join('./DIV2K_train_LR_bicubic/X{}_sub/'.format(self.scale), self.img_list[index]))
            hr_img = cv2.imread(os.path.join('./DIV2K_train_HR_sub/', self.img_list[index]))

            if np.random.rand() < 0.2:
                # without any augmentation.
                lr_img = tfs.ToTensor()(lr_img)
                hr_img = tfs.ToTensor()(hr_img)
                return lr_img, hr_img
        else:
            lr_img = cv2.imread(os.path.join('./DIV2K_valid_LR_bicubic/X{}_sub/'.format(self.scale), self.img_list[index]))
            hr_img = cv2.imread(os.path.join('./DIV2K_valid_HR_sub/', self.img_list[index]))
            lr_img = tfs.ToTensor()(lr_img)
            hr_img = tfs.ToTensor()(hr_img)
            return lr_img, hr_img

        lr_img = Image.fromarray(lr_img, mode='RGB')
        hr_img = Image.fromarray(hr_img, mode='RGB')

        trans1 = tfs.RandomHorizontalFlip(p=0.5)
        trans2 =  tfs.RandomRotation(degrees=90)
        trans3 =  tfs.RandomRotation(degrees=180)
        trans4 =  tfs.RandomRotation(degrees=270)
        choose_trans = tfs.RandomChoice([trans1, trans2, trans3, trans4])
        
        lr_img = choose_trans(lr_img)
        hr_img = choose_trans(hr_img)

        lr_img = tfs.ToTensor()(lr_img)
        hr_img = tfs.ToTensor()(hr_img)
        
        return lr_img, hr_img

    def __len__(self):
        return len(self.img_list)
    
