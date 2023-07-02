import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import RCAN
from dataset import DIV2K
import torch
import os
from utils import AverageMeter, calc_psnr, calc_ssim
import cv2
import numpy as np

lossFunc = torch.nn.L1Loss()

def train(epoch, model, optimizer, scheduler, train_loader, writer):
    # the number of iterations have been done.
    iteration = len(train_loader) * epoch
    losses = AverageMeter()
    psnr = AverageMeter()
    ssim = AverageMeter()
    for lr_img, hr_img in tqdm(train_loader):
        bsz = lr_img.shape[0]
        if torch.cuda.is_available():
            lr_img = lr_img.cuda()
            hr_img = hr_img.cuda()

        optimizer.zero_grad()
        output = model(lr_img)
        loss = lossFunc(output, hr_img)
        psnr_batch = calc_psnr(output, hr_img)
        ssim_batch = calc_ssim(output, hr_img)

        # update metric
        losses.update(loss.item(), bsz) 
        psnr.update(psnr_batch, bsz)
        ssim.update(ssim_batch, bsz)

        loss.backward()
        optimizer.step()
        scheduler.step()

        iteration += 1
        if iteration % 50 == 0:
            writer.add_scalar('train/loss', losses.val, iteration)
            writer.add_scalar('train/psnr', psnr.val, iteration)
            writer.add_scalar('train/ssim', ssim.val, iteration)


    return 


def validate(epoch, model, val_loader, writer):
    model.eval()
    psnr = AverageMeter()
    ssim = AverageMeter()
    for lr_img, hr_img in tqdm(val_loader):
        bsz = lr_img.shape[0]
        if torch.cuda.is_available():
            lr_img = lr_img.cuda()
            hr_img = hr_img.cuda()
                
        output = model(lr_img)
        psnr_batch = calc_psnr(output, hr_img)
        ssim_batch = calc_ssim(output, hr_img)
        # update metric
        psnr.update(psnr_batch, bsz)
        ssim.update(ssim_batch, bsz)


    writer.add_scalar('valid/psnr', psnr.val, epoch)
    writer.add_scalar('valid/ssim', ssim.val, epoch)



def run(args):
    save_folder = os.path.join('./exps', args.exp_name)
    ckpt_folder = os.path.join(save_folder, 'ckpt')
    log_folder = os.path.join(save_folder, 'log')
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)
    writer = SummaryWriter(log_dir=log_folder)
    
    # define dataset and dataloader
    train_dataset = DIV2K(scale=args.scale, train=True)
    val_dataset = DIV2K(scale=args.scale, train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = RCAN(args)
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_schedule, gamma=0.5)

    if args.cont:
        # load latest checkpoint
        ckpt_lst = os.listdir(ckpt_folder)
        ckpt_lst.sort(key=lambda x: int(x.split('_')[-1]))
        read_path = os.path.join(ckpt_folder, ckpt_lst[-1])
        print('load checkpoint from %s'%(read_path))
        checkpoint = torch.load(read_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.total_epoch):
        train(epoch, model, optimizer, scheduler, train_loader, writer)
        
        if epoch % args.save_freq == 0:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(ckpt_folder, 'ckpt_epoch_%s'%(str(epoch)))
            torch.save(state, save_file)

        with torch.no_grad():
            validate(epoch, model, val_loader, writer)
    return 

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--n_RG', '-rg', type=int, default=10, help='the number of RwsiudalGroup Blocks')
    arg_parser.add_argument('--n_RCAB', '-rcab', type=int, default=20, help='the number of RCAB blocks')
    arg_parser.add_argument('--n_feat', '-n', type=int, default=64, help='the number of feature maps')
    arg_parser.add_argument('--kernel_size', '-kz', type=int, default=3, help='kernel size')
    arg_parser.add_argument('--reduction', '-r', type=int, default=16, help='reduction ratio of Channel Attention')
    arg_parser.add_argument('--scale', '-s', type=int, default=2, help='upscale factor')
    arg_parser.add_argument('--bias', '-bias', type=bool, default=False, help='whether to use bias in Conv layers')
    arg_parser.add_argument('--bn', '-bn', type=bool, default=False, help='whether to use batchnorm in model')
    arg_parser.add_argument('--lr', '-lr', type=float, default=1e-4, help='learning rate')
    arg_parser.add_argument('--lr_schedule', '-lr_s', type=float, default=2e5, help='step size of learning rate scheduler')
    arg_parser.add_argument('--batch_size', '-bz', type=int, default=16, help='batch size')
    arg_parser.add_argument('--save_freq', '-save', type=int, default=1, help='frequency of saving model')
    arg_parser.add_argument('--total_epoch', '-t', type=int, default=10, help='total epoch number for training')
    arg_parser.add_argument('--exp_name', '-name', type=str, required=True, help='the checkpoints and logs will be saved in ../exps/$EXP_NAME/')
    arg_parser.add_argument('--cont', '-c', action='store_true', help="whether to load saved checkpoints from $EXP_NAME and continue training")

    args = arg_parser.parse_args()

    run(args)