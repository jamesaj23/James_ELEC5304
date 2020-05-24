# -*- coding: utf-8 -*-

# PyTorch 0.4.1, https://pytorch.org/docs/stable/index.html

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26}, 
#    number={7}, 
#    pages={3142-3155}, 
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/SaoYan/DnCNN-PyTorch
# =============================================================================

# run this to train the model

# =============================================================================
# For batch normalization layer, momentum should be a value from [0.1, 1] rather than the default 0.1. 
# The Gaussian noise output helps to stablize the batch normalization, thus a large momentum (e.g., 0.95) is preferred.
# =============================================================================

# import necessary packages
import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import data_generator as dg
from data_generator import DenoisingDataset
# ============================================
# import network from model.py under the same folder where this file is.
from model import DnCNN
# ============================================


# Params
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=20, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
args = parser.parse_args()
args.sigma = 25
args.train_data = 'data/Train400'

sigma = args.sigma
save_dir = os.path.join('models', args.model+'_' + 'sigma' + str(sigma))
cuda = torch.cuda.is_available()
batch_size = args.batch_size
n_epoch = args.epoch

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


if __name__ == '__main__':
    print('===> Building model')
    model = DnCNN(depth=5)
    if cuda:
        model = model.cuda()
    model.train()

    initial_epoch = 0
    criterion = sum_squared_error()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates

    for epoch in range(initial_epoch, n_epoch):

        scheduler.step(epoch)  # step to the learning rate in this epcoh
        xs = dg.datagenerator(data_dir=args.train_data)
        xs = xs.astype('float32')/255.0
        xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
        DDataset = DenoisingDataset(xs, sigma)
        DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        start_time = time.time()

        for n_count, batch_yx in enumerate(DLoader):
                optimizer.zero_grad()
                if cuda:
                    batch_x, batch_y = batch_yx[1].cuda(), batch_yx[0].cuda()
                else:
                	batch_x, batch_y = batch_yx[1], batch_yx[0]
                loss = criterion(model(batch_y), batch_x)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                if n_count % 10 == 0:
                    print('%4d %4d / %4d loss = %2.4f' % (epoch+1, n_count, xs.size(0)//batch_size, loss.item()/batch_size))
        elapsed_time = time.time() - start_time

        log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch+1, epoch_loss/n_count, elapsed_time))
        np.savetxt('train_result.txt', np.hstack((epoch+1, epoch_loss/n_count, elapsed_time)), fmt='%2.4f')
        # torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
