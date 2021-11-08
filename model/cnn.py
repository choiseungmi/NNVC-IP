import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import numpy as np

import config
import os
import math

use_cuda = torch.cuda.is_available()
dtype = 'float32' if use_cuda else 'float64'
torchtype = {'float32': torch.float32, 'float64': torch.float64}

qp_list = ['22', '27', '32', '37']
block_len = [4, 8, 16, 32, 64]
block_size = {
    4:{
        4:[(4,12),(8,4)],
        8:[(4,20),(8,4)],
        16:[(4,36),(8,4)],
        32:[(4,68),(8,4)]
    },
    8:{
        4:[(4,12),(16,4)],
        8:[(8,24),(16,8)],
        16:[(8,40),(16,8)],
        32:[(8,72),(16,8)]
    },
    16:{
        4:[(4,12),(32,4)],
        8:[(8,24),(32,8)],
        16:[(8,40),(32,8)],
        32:[(8,80),(32,16)]
    },
    32:{
        4: [(4, 12), (64, 4)],
        8: [(8, 24), (64, 8)],
        16: [(16, 40), (64, 8)],
        32: [(16, 80), (64, 16)]
    },
    64:{
        64:[(32,160),(128,32)]
    }
}


def fully_module(in_num, out_num):
    return nn.Sequential(
        nn.Linear(in_num, out_num),
        nn.LeakyReLU())

def conv_module(in_num, out_num, stride):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())

class CNNBaseBlock(nn.Module):
    def __init__(self, block_size, stride):
        super(CNNBaseBlock, self).__init__()
        self.block_size = block_size
        self.layer1 = conv_module(1, 32, (2, 2))
        self.layer2 = conv_module(32, 64, (2, 2))
        self.layer3 = conv_module(64, 128, stride)
        self.layer4 = conv_module(128, 128, stride)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(-1, self.block_size)
        return out


class FNBaseBlock(nn.Module):
    def __init__(self,  in_num, h, w):
        super(FNBaseBlock, self).__init__()
        self.layer1 = fully_module(in_num, 1200)
        self.layer2 = fully_module(1200, 1200)
        self.layer3 = nn.Linear(1200, h*w)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


class TextureAdaptivePNN(nn.Module):
    def __init__(self, is_fully_connected, h, w):
        super(TextureAdaptivePNN, self).__init__()
        self.is_fully_connected = is_fully_connected
        above_size, left_size = block_size[h][w]
        if self.is_fully_connected:
            self.full_size = above_size[0] * above_size[1] + left_size[0] * left_size[1]
            self.linear = FNBaseBlock(self.full_size, h, w)
        else:
            self.above_out_size = math.ceil(above_size[0]/4)*math.ceil(above_size[1]/16)*128
            self.left_out_size = math.ceil(left_size[0]/16)*math.ceil(left_size[1]/4)*128
            self.above_layer = CNNBaseBlock(self.above_out_size, (1, 2))
            self.left_layer = CNNBaseBlock(self.left_out_size, (2, 1))
            self.linear1 = fully_module(self.above_out_size+self.left_out_size, h*w)
            self.linear2 = fully_module(h*w, h*w)

    def forward(self, above, left):
        if self.is_fully_connected:
            batch = above.shape[0]
            above = above.view(batch, -1)
            left = left.view(batch, -1)
            vector = torch.cat((above, left), dim=1)
            out = self.linear(vector)
        else:
            out_above = self.above_layer(above)
            out_left = self.left_layer(left)
            out = torch.cat((out_above, out_left), dim=1)
            out = self.linear1(out)
            out = self.linear2(out)
        return out

class TextureAdaptiveCluster(nn.Module):
    def __init__(self, is_fully_connected, h, w):
        super(TextureAdaptivePNN, self).__init__()
        self.is_fully_connected = is_fully_connected
        above_size, left_size = block_size[h][w]
        if self.is_fully_connected:
            self.full_size = above_size[0] * above_size[1] + left_size[0] * left_size[1]
            self.linear = FNBaseBlock(self.full_size, h, w)
        else:
            self.above_out_size = math.ceil(above_size[0]/4)*math.ceil(above_size[1]/16)*128
            self.left_out_size = math.ceil(left_size[0]/16)*math.ceil(left_size[1]/4)*128
            self.above_layer = CNNBaseBlock(self.above_out_size, (1, 2))
            self.left_layer = CNNBaseBlock(self.left_out_size, (2, 1))
            self.linear1 = fully_module(self.above_out_size+self.left_out_size, h*w)
            self.linear2 = fully_module(h*w, h*w)

    def forward(self, above, left):
        if self.is_fully_connected:
            batch = above.shape[0]
            above = above.view(batch, -1)
            left = left.view(batch, -1)
            vector = torch.cat((above, left), dim=1)
            out = self.linear(vector)
        else:
            out_above = self.above_layer(above)
            out_left = self.left_layer(left)
            out = torch.cat((out_above, out_left), dim=1)
            out = self.linear1(out)
            out = self.linear2(out)
        return out


if __name__ == '__main__':
    train_path = config.train_numpy_path
    print(use_cuda)
    h = block_len[3]
    w = block_len[3]
    above_numpy_path = os.path.join(train_path, qp_list[1], str(h)+"x"+str(w), "above")
    above_sequence_list = os.listdir(above_numpy_path)
    left_numpy_path = os.path.join(train_path, qp_list[1], str(h) + "x" + str(w), "left")
    left_sequence_list = os.listdir(left_numpy_path)
    above_np = np.empty((0, 1, block_size[h][w][0][0], block_size[h][w][0][1]), np.int8)
    left_np = np.empty((0, 1, block_size[h][w][1][0], block_size[h][w][1][1]), np.int8)
    for i, a_path in enumerate(above_sequence_list):
        if i==5:break
        a = np.load(os.path.join(above_numpy_path, a_path))
        above_np = np.append(above_np, np.reshape(a, ((1,)+(1,)+a.shape)), axis=0)
    for i, l_path in enumerate(left_sequence_list):
        if i==5:break
        l = np.load(os.path.join(left_numpy_path, l_path))
        left_np = np.append(left_np, np.reshape(l, ((1,)+(1,)+l.shape)), axis=0)

    above = torch.from_numpy(above_np).float()
    left = torch.from_numpy(left_np).float()
    print(above.shape)
    print(left.shape)
    batch = above.shape[0]
    vector = torch.cat((above.view(batch, -1), left.view(batch, -1)), dim=1)
    print(vector.shape[1])

    model = TextureAdaptivePNN(False, h, w)
    model.cuda()
    model(above.cuda(), left.cuda())