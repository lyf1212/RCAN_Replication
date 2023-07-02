import torch 
import torch.nn as nn
import numpy as np
import math
import cv2
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _psnr(img1, img2):
    # img1 and img2 have range [0, 255], uint8, (C, H, W).

    img = cv2.cvtColor(img1.transpose(1, 2, 0), cv2.COLOR_BGR2YCR_CB)
    img_ = cv2.cvtColor(img2.transpose(1, 2, 0), cv2.COLOR_BGR2YCR_CB)
    # calculate psnr and ssim on Y channel of transformed YCbCr space.
    return peak_signal_noise_ratio(img[..., 0], img_[..., 0], data_range=1)


def _ssim(img1, img2):
    # img1 and img2 have range [0, 255], uint8, (C, H, W).
    img = cv2.cvtColor(img1.transpose(1, 2, 0), cv2.COLOR_BGR2YCR_CB)
    img_ = cv2.cvtColor(img2.transpose(1, 2, 0), cv2.COLOR_BGR2YCR_CB)
    # calculate psnr and ssim on Y channel of transformed YCbCr space.
    return structural_similarity(img[...,0], img_[...,0], win_size=11, data_range=1)


def calc_psnr(img1_batch, img2_batch):
    res = []
    for i, img1 in enumerate(img1_batch):   
        res.append(_psnr(img1.cpu().detach().numpy(), img2_batch[i].cpu().detach().numpy()))
    return np.mean(np.array(res))



def calc_ssim(img1_batch, img2_batch):
    res = []
    for i, img1 in enumerate(img1_batch):   
        res.append(_ssim(img1.cpu().detach().numpy(), img2_batch[i].cpu().detach().numpy()))
    return np.mean(np.array(res))


# def collate_wrapper(batch):
    '''
    wrapper function of collate_fn to get the same size of data.
    Use 'extract_subimages' instead of this memory-expensive method.
    '''
    
#     max_H1 = 0
#     max_W1 = 0
#     max_H2 = 0
#     max_W2 = 0
#     lr_imgs = []
#     hr_imgs = []
#     for img1, img2 in batch:
#         max_H1 = max(img1.size()[1], max_H1)
#         max_W1 = max(img1.size()[2], max_W1)
#         max_H2 = max(img2.size()[1], max_H2)
#         max_W2 = max(img2.size()[2], max_W2)
#     for img1, img2 in batch:
#         img1 = img1.unsqueeze(0)
#         img2 = img2.unsqueeze(0)

#         img1 = F.pad(img1, \
#                      (math.floor((max_W1 - img1.size()[3])/2), math.ceil((max_W1 - img1.size()[3])/2),\
#                       math.floor((max_H1 - img1.size()[2])/2), math.ceil((max_H1 - img1.size()[2])/2)), 'reflect')
#         img2 = F.pad(img2, \
#                      (math.floor((max_W2 - img2.size()[3])/2), math.ceil((max_W2 - img2.size()[3])/2), \
#                       math.floor((max_H2 - img2.size()[2])/2), math.ceil((max_H2 - img2.size()[2])/2)), 'reflect')
#         lr_imgs.append(img1.squeeze(0))
#         hr_imgs.append(img2.squeeze(0))
#     return torch.stack(lr_imgs, 0), torch.stack(hr_imgs, 0)


