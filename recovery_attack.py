import os
import numpy as np
import torch
import sys
sys.path.append('.')
from utils.utils import compute_batch_recovery_metric, mkdir_if_missing, setup_logger
import torch.optim as optim
from modeling.generator import define_G
from torchvision import utils as v_utils

import argparse
parser = argparse.ArgumentParser(description='recovery attack with UNet')
parser.add_argument('--npy_path', type=str, default='')
parser.add_argument('--gpus', type=list, default=[0,1],help='appoint GPU devices')
parser.add_argument('--initial_lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--total_epochs', type=int, default=100, help='total epochs')
# training config
parser.add_argument("--generator", type=str, default='unet_128')
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--debug", action='store_true', default=False)
args = parser.parse_args()

#set logger
log_dir = os.path.dirname(args.npy_path)
logger = setup_logger("recovery", log_dir, 0, save_name='log_recovery.txt')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# build recovery network
Generator = define_G(input_nc=3, output_nc=3, ngf=64, netG=args.generator).to(device)

Loss_l1 = torch.nn.L1Loss()
Generator_optimizer = optim.Adam(Generator.parameters(), lr=args.lr)

data = os.listdir(args.npy_path)
data.sort()   # important code

train_ori_datas = [os.path.join(args.npy_path, d) for d in data if 'clean-train' in d and 'images' in d]
train_enc_datas = [os.path.join(args.npy_path, d) for d in data if 'encry-train' in d and 'images' in d]
test_ori_datas = [os.path.join(args.npy_path, d) for d in data if 'clean-test' in d and 'images' in d]
test_enc_datas = [os.path.join(args.npy_path, d) for d in data if 'encry-test' in d and 'images' in d]

if len(test_ori_datas)==0 or len(train_enc_datas)==0 or len(test_ori_datas)==0 or len(test_enc_datas)==0:
    assert ValueError, "please check your datas"

# make sure each ori_data and enc_data are pairs of the same ID
for i in range(len(train_ori_datas)):
    train_ori = train_ori_datas[i]
    train_enc = train_enc_datas[i]
    if train_ori.replace("clean-train","encry-train") != train_enc:
        raise ValueError("Please make sure each ori_data and enc_data are pairs of the same ID")
    test_ori = test_ori_datas[i]
    test_enc = test_enc_datas[i]
    if test_ori.replace("clean-test","encry-test") != test_enc:
        raise ValueError("Please make sure each ori_data and enc_data are pairs of the same ID")


def normalize(noise, max_num=255, min_num=0):
    noise = (noise - noise.min()) / (noise.max() - noise.min()) * (max_num - min_num) + min_num
    return noise

best_ssim = 0.0
lowest_ssim = 100.0
logger.info("---------Start Training---------")
for epoch in range(args.total_epochs):
    concat_ori_images = None
    # for i, (ori_data, enc_data) in enumerate(zip(ori_datas, enc_datas)):
    Generator.train()
    for i, (ori_data, enc_data) in enumerate(zip(train_ori_datas, train_enc_datas)):
        ori_images = torch.from_numpy(np.load(ori_data)).cuda()
        enc_images = torch.from_numpy(np.load(enc_data)).cuda()

        output = Generator(enc_images)
        loss = Loss_l1(output, ori_images)

        Generator_optimizer.zero_grad()
        loss.backward()
        Generator_optimizer.step()
        concat_ori_images = None
        if i%20==0:
            logger.info("Generator [%d/%d] [iter:%d] loss: %0.6f " % (epoch, args.total_epochs, i, loss.item()))

    # recovery attack on test images
    if epoch==2 or epoch==args.total_epochs-1:
        logger.info("---------Start Testing---------")
        Generator.eval()
        with torch.no_grad():
            total_num = 0
            ori_psnrs, ori_mses, ori_ssims = 0.0, 0.0, 0.0
            psnrs, mses, ssims = 0.0, 0.0, 0.0
            for idx, (label, enc_data) in enumerate(zip(test_ori_datas, test_enc_datas)):
                labels = torch.from_numpy(np.load(label)).cuda()
                enc_images = torch.from_numpy(np.load(enc_data)).cuda()

                bz = labels.shape[0]
                total_num += bz
                rec_images = Generator(enc_images)
                if rec_images.shape != labels.shape:
                    raise ValueError

                # compute metric
                rec_images, enc_images, labels = rec_images.cpu().detach(), enc_images.cpu().detach(), labels.cpu().detach()
                normalized_labels = (labels - labels.min()) / (labels.max() - labels.min())
                normalized_rec_images = (rec_images - rec_images.min()) / (rec_images.max() - rec_images.min())
                normalized_enc_images = (enc_images - enc_images.min()) / (enc_images.max() - enc_images.min())
                ori_psnr, ori_mse, ori_ssim = compute_batch_recovery_metric(normalized_enc_images, normalized_labels)
                psnr, mse, ssim = compute_batch_recovery_metric(normalized_rec_images, normalized_labels)
                ori_psnrs, ori_mses, ori_ssims = ori_psnrs+ori_psnr*bz, ori_mses+ori_mse*bz, ori_ssims+ori_ssim*bz
                psnrs, mses, ssims = psnrs+psnr*bz, mses+mse*bz, ssims+ssim*bz
                if epoch==args.total_epochs-1 and idx<20:
                    visuals = torch.cat([enc_images, rec_images, labels], dim=0)
                    save_path = os.path.join(log_dir,'rec_images')
                    mkdir_if_missing(save_path)
                    v_utils.save_image(visuals, os.path.join(save_path, '{}.png'.format(idx)), normalize=True, padding=0, nrow=bz)

        logger.info("--------protection metrics--------")
        logger.info("***protected_images***:")
        logger.info("psnr:{:.2f}  mse:{:.4f}  ssim:{:.2f}".format(ori_psnrs/total_num, ori_mses/total_num, ori_ssims/total_num))
        logger.info("***recovered_images***:")
        logger.info("psnr:{:.2f}  mse:{:.4f}  ssim:{:.2f}".format(psnrs/total_num, mses/total_num, ssims/total_num))
        
        torch.save(Generator.state_dict(), os.path.join(log_dir,'Generator_weight.pth'))
            
        Generator.train()
