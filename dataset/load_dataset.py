import glob
import os
from typing import Optional, Callable, Tuple, Dict, Any, List

import torch
import torchvision
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from models.tapnn import block_size

def write_list(above_path, left_path, y_path, h, w):
    file_path = os.path.join(y_path[:-2], 'seq.txt')
    if os.path.exists(file_path): return 0
    return_path = []
    above_size = block_size[h][w][0]
    left_size = block_size[h][w][1]
    y_size = (h, w)
    error_num = 0
    for seq in tqdm(sorted(os.listdir(y_path))):
        above = np.int8(np.load(os.path.join(above_path, seq)))
        left = np.int8(np.load(os.path.join(left_path, seq)))
        y = np.int8(np.load(os.path.join(y_path, seq)))
        if above_size != above.shape or left_size != left.shape or y_size!=y.shape:
            error_num+=1
            continue
        return_path.append(seq)
    print(error_num)
    with open(file_path, 'w') as f:
        for i in return_path:
            f.write(i+"\n")

def check_list(y_path, is_test):
    with open(os.path.join(y_path[:-2], "seq.txt"), "r") as f:
        return_path = f.read().splitlines()
    # if is_test:
    #     return return_path[:500]
    # return return_path[500:]
    return return_path

def check_list_cluster(y_path, k):
    with open(os.path.join(y_path[:-2], "seq_"+str(k)+".txt"), "r") as f:
        return_path = f.read().splitlines()
    return return_path

def prepare_image(path_to_files, h, w, km):
    above_path = os.path.join(path_to_files, "above")
    left_path = os.path.join(path_to_files, "left")
    y_path = os.path.join(path_to_files, "y")

    if os.path.exists(os.path.join(y_path[:-2], "seq_"+str(0)+".txt")): return 0

    write_list(above_path, left_path, y_path, h, w)
    samples = sorted(check_list(y_path, False))

    fn_imgs = []
    for seq in samples:
        above = np.int8(np.load(os.path.join(above_path, seq)))
        left = np.int8(np.load(os.path.join(left_path, seq)))
        picture = np.zeros((above.shape[1], above.shape[1]), np.uint8)
        picture[:above.shape[0], :above.shape[1]] = above
        picture[above.shape[0]:, :above.shape[0]] = left
        fn_imgs.append(picture)

    imgs_dict = np.array(fn_imgs)
    nsamples, nx, ny = imgs_dict.shape
    imgs_dict = imgs_dict.reshape((nsamples, nx * ny))
    predict = km.predict(imgs_dict)

    for i, k in enumerate(predict):
        with open(os.path.join(y_path[:-2], "seq_"+str(k)+".txt"), 'a') as f:
            f.write(samples[i]+"\n")

class TAPNN(Dataset):
    def __init__(self, root, h, w, transform, km, k):
        self.root = root
        self.above_path = os.path.join(self.root, "above")
        self.left_path = os.path.join(self.root, "left")
        self.y_path = os.path.join(self.root, "y")

        prepare_image(self.root, h, w, km)
        self.samples = check_list_cluster(self.y_path, k)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        seq = self.samples[idx]
        above = np.int8(np.load(os.path.join(self.above_path, seq)))
        left = np.int8(np.load(os.path.join(self.left_path, seq)))
        y = np.int8(np.load(os.path.join(self.y_path, seq)))

        above = self.transform(above).float()
        left = self.transform(left).float()
        y = self.transform(y).float()

        return above, left, y

class TAPNN_pred(Dataset):
    def __init__(self, root, h, w, transform, km, k):
        self.root = root
        self.above_path = os.path.join(self.root, "above")
        self.left_path = os.path.join(self.root, "left")
        self.y_path = os.path.join(self.root, "y")
        self.pred_path = os.path.join(self.root, "pred")

        prepare_image(self.root, h, w, km)
        self.samples = check_list_cluster(self.y_path, k)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, str]:
        seq = self.samples[idx]
        above = np.int8(np.load(os.path.join(self.above_path, seq)))
        left = np.int8(np.load(os.path.join(self.left_path, seq)))
        y = np.int8(np.load(os.path.join(self.y_path, seq)))
        pred = np.int8(np.load(os.path.join(self.pred_path, seq)))

        above = self.transform(above).float()
        left = self.transform(left).float()
        y = self.transform(y).float()
        pred = self.transform(pred).float()

        return above, left, y, pred, self.samples[idx]

class ClusterDataset(Dataset):
    def __init__(self, root, h, w, transform):
        self.root = root
        self.above_path = os.path.join(self.root, "above")
        self.left_path = os.path.join(self.root, "left")
        self.y_path = os.path.join(self.root, "y")

        write_list(self.above_path, self.left_path, self.y_path, h, w)
        self.samples = check_list(self.y_path, False)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        seq = self.samples[idx]
        above = np.int8(np.load(os.path.join(self.above_path, seq)))
        left = np.int8(np.load(os.path.join(self.left_path, seq)))
        picture = np.zeros((above.shape[1], above.shape[1]), np.uint8)
        picture[:above.shape[0],:above.shape[1]] = above
        picture[above.shape[0]:, :above.shape[0]] = left

        picture = self.transform(picture).float()
        pseudolabels = [idx] * 8

        return picture, pseudolabels