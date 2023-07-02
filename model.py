import torch.nn as nn
import math
import torch

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            # Use 'bias=True' to increase capability of the model, but maybe useless when use batchnorm at the same time.
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),    
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        compressed_x = self.avg_pool(x)
        s = self.conv(compressed_x)
        return s * x
    

class RCAB(nn.Module):
    def __init__(self, n_feat, reduction, kernel_size, bias, bn):
        super().__init__()
        blk = []
        for i in range(2):
            blk.append(nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, padding=1, bias=bias))
            if bn:
                blk.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                blk.append(nn.ReLU(inplace=True))
        blk.append(ChannelAttention(n_feat, reduction))
        self.conv = nn.Sequential(*blk)
    def forward(self, x):
        return x + self.conv(x)


class RG(nn.Module):
    def __init__(self, n_RCAB, n_feat, reduction, kernel_size, bias, bn):
        super().__init__()
        blk = []
        for i in range(n_RCAB):
            blk.append(RCAB(n_feat, reduction, kernel_size, bias, bn))
        blk.append(nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, padding=1, bias=bias))
        self.conv = nn.Sequential(*blk)
    def forward(self, x):
        return x + self.conv(x)
    
class Upscale(nn.Module):
    def __init__(self, scale, n_feat, kernel_size, bias, bn, nonlinear=False):
        super().__init__()
        blk = []
        # upscale the feature map x2 a time if scale == 2^n, or directly upscale if scale==3, for obtaining detail information better.
        # cannot handle scale=5, 7, 9 or bigger just like that.
        if (scale & (scale - 1)) == 0:
            # Judge scale is 2^n.
            # If scale == 2^n, there is only one '1' at first and all '0' after that, which can be distinguished by -1.
            for i in range(int(math.log(scale, 2))):
                blk.append(nn.Conv2d(n_feat, n_feat * 4, kernel_size, padding=1, bias=bias))
                # nn.PixelShuffle(upscale_factor):
                # (B, C*r^2, H, W)  ->  (B, C, H*r, W*r), where r is upscale_factor.
                blk.append(nn.PixelShuffle(2))
                if bn:
                    blk.append(nn.BatchNorm2d(n_feat))
                if nonlinear:
                    blk.append(nn.ReLU())
        elif scale == 3:
            blk.append(nn.Conv2d(n_feat, n_feat * 9, kernel_size, padding=1, bias=bias))
            blk.append(nn.PixelShuffle(3))
            if bn:
                blk.append(nn.BatchNorm2d(n_feat))
            if nonlinear:
                blk.append(nn.ReLU())
        else:
            raise NotImplementedError
        self.upscale_pipeline = nn.Sequential(*blk)
    def forward(self, x):
        return self.upscale_pipeline(x)

class RCAN(nn.Module):
    def __init__(self, args):
        super().__init__()
        n_RG = args.n_RG
        n_RCAB = args.n_RCAB
        n_feat = args.n_feat
        kernel_size = args.kernel_size
        reduction = args.reduction

        self.shallow_feature = nn.Conv2d(3, n_feat, kernel_size=kernel_size, padding=1)
        
        RIR_blk = []
        for i in range(n_RG):
            RIR_blk.append(RG(n_RCAB, n_feat, reduction, kernel_size, args.bias, args.bn))
        RIR_blk.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=1))
        self.RIR = nn.Sequential(*RIR_blk)
        self.upscale = Upscale(args.scale, n_feat, kernel_size, args.bias, args.bn, nonlinear=False)
        
        self.tail = nn.Conv2d(n_feat, 3, kernel_size, padding=1)
        
        
    def forward(self, x):
        y = self.shallow_feature(x)
        z = self.RIR(y)
        u = z + y
        res = self.tail(self.upscale(u))
        return res





        