import argparse
import random
import shutil
import sys
import time
from tkinter import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import numpy as np
import config

from cluster import load_cluster
from models.tapnn import TextureAdaptivePNN, block_size
from dataset.load_dataset import TAPNN, TAPNN_pred
from util import AverageMeter, CSVLogger

import pandas as pd

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, help="Training dataset"
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=0,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=22,
        help="Quality (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument('--clusterk', type=int, default=0, help='cluster index (default: 0)')
    parser.add_argument("-hgt", "--height", type=int, default=32, help="block height")
    parser.add_argument("-wdt", "--width", type=int, default=32, help="block width")
    args = parser.parse_args(argv)
    return args

def test_epoch(test_dataloader, model, criterion, road_file_path, k, save_img_path):
    model.eval()
    device = next(model.parameters()).device
    loss = AverageMeter()
    pred_loss = AverageMeter()

    road_df = pd.read_csv(road_file_path)

    with torch.no_grad():
        for (a, l, y, pred, filename) in test_dataloader:
            a = a.to(device)
            l = l.cuda()
            y = y.cuda()
            pred = pred.cuda()
            out = model(a, l)
            out = out.view(y.shape)
            filename = filename[0]

            avg = road_df.loc[road_df.files == filename]['avg'].values[0]
            out_criterion = criterion(out, y)
            pred_criterion = criterion(pred, y)

            road_df.loc[road_df.files == filename, 'loss'] = out_criterion.item()

            road_df.loc[road_df.files == filename, 'pred_loss'] = pred_criterion.item()

            road_df.loc[road_df.files == filename, 'k'] = k
            road_df.to_csv(road_file_path, index=False)

            loss.update(out_criterion)
            pred_loss.update(pred_criterion)

            save_image(save_img_path, filename[:-4]+".png", y, pred, out)

    del a
    del l
    del y
    # GPU memory delete
    torch.cuda.empty_cache()
    return loss.avg.item(), pred_loss.avg.item()

def main(argv):
    args = parse_args(argv)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    ############################################### load dataset ##############################################
    if args.dataset is None:
        args.dataset = config.train_numpy_path
    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )
    path_model_cluster = os.path.join(config.cluster_checkpoint, str(args.height)+"x"+str(args.width) + "_" + str(args.quality) + '.pkl')
    km = load_cluster(path_model_cluster)

    test_dataset = TAPNN_pred(os.path.join(config.inference_numpy_path, str(args.quality), str(args.height)+"x"+str(args.width)), args.height, args.width,
                         transform=test_transforms, km=km, k=args.clusterk)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(torch.cuda.is_available())
    print(device)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    ############################################## model #####################################################
    h = args.height
    w = args.width

    model = TextureAdaptivePNN(False, h, w)
    model.cuda()

    criterion = nn.MSELoss()

    checkpoint = torch.load("checkpoint\\best_loss_" +str(args.height)+"x"+str(args.width)+"_" + str(args.quality) +"_" + str(args.clusterk)+".pth.tar", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    start = time.time()
    dp_path = os.path.join(config.inference_numpy_path, str(args.quality), str(args.height)+"x"+str(args.width), "seq_info.csv")
    img_path = os.path.join(config.inference_numpy_path, str(args.quality), str(args.height)+"x"+str(args.width), "image")

    loss, pred_loss = test_epoch(test_dataloader, model, criterion, dp_path, args.clusterk, img_path)

    print(f'\tTest Loss: {loss:.3f}'
          f'\tPred Loss: {pred_loss:.3f}')

    print(f"Total TIme: {time.time() - start}")

def save_image(root, seq, y, pred, out):

    y_path = os.path.join(root, "y", seq)
    pred_path = os.path.join(root, "pred", seq)
    out_path = os.path.join(root, "out", seq)

    torchvision.utils.save_image(y, y_path)
    torchvision.utils.save_image(pred, pred_path)
    torchvision.utils.save_image(out, out_path)

    # y = np.reshape(y, (33, 33))
    # im = Image.fromarray(y)
    # im.save(y_path)
    # y = np.reshape(pred, (33, 33))
    # im = Image.fromarray(y)
    # im.save(pred_path)
    # y = np.reshape(out, (33, 33))
    # im = Image.fromarray(y)
    # im.save(out_path)



if __name__ == "__main__":
    main(sys.argv[1:])