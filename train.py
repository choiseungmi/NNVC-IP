import argparse
import math
import random
import shutil
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import csv
import cv2
import numpy as np


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    # mse 함수를 4:1:1로 바꾸기
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        o_y, o_cb, o_cr = output["x_hat"].chunk(3, -3)
        t_y, t_cb, t_cr = target.chunk(3, -3)
        mse_y = self.mse(o_y, t_y)
        mse_cb = self.mse(o_cb, t_cb)
        mse_cr = self.mse(o_cr, t_cr)

        out["mse_loss"] = (4 * mse_y + mse_cb + mse_cr) / 6
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    parameters = set(
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    )
    aux_parameters = set(
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    )
    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters
    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0
    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(list(parameters))),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(list(aux_parameters))),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
        model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    a_aux_loss = AverageMeter()
    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        out_net = model(d)
        out_criterion = criterion(out_net, d)

        bpp_loss.update(out_criterion["bpp_loss"])
        loss.update(out_criterion["loss"])
        mse_loss.update(out_criterion["mse_loss"])

        out_criterion["loss"].backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        aux_loss = model.aux_loss()
        a_aux_loss.update(aux_loss)
        aux_loss.backward()
        aux_optimizer.step()
        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i * len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )
    return loss.avg, bpp_loss.avg, a_aux_loss.avg


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)
            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )
    return loss.avg, bpp_loss.avg, aux_loss.avg


def save_checkpoint(state, is_best, q, filename="checkpoint"):
    torch.save(state, filename + q + ".pth.tar")
    if is_best:
        shutil.copyfile(filename + q + ".pth.tar", "checkpoint_best_loss" + q + ".pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-hyperprior",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=0,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=3,
        help="Quality (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--save", action="store_true", help="Save model to disk")
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


class CSVLogger():
    def __init__(self, fieldnames, filename='log.csv'):
        self.filename = filename
        self.csv_file = open(filename, 'a')
        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)

    # self.writer.writeheader()
    # self.csv_file.flush()
    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()


class Blur(object):
    def __init__(self, k, sig):
        self.k = k
        self.sig = sig

    def __call__(self, img):
        r = np.random.rand(1)
        if r < 0.5:
            img = cv2.GaussianBlur(img.numpy(), (self.k, self.k), self.sig)
            img = torch.from_numpy(img)
        return img


class RGB2YCbCr(object):
    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        img = rgb2ycbcr(img)
        return img


def main(argv):
    args = parse_args(argv)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size),  # 이미지 크기 조절
         transforms.RandomRotation(30),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()]  # numpy이미지에서 torch이미지로 변경
    )
    train_transforms.transforms.append(RGB2YCbCr())

    #     print(train_transforms.shape)
    #     train_transforms=rgb2ycbcr(train_transforms)
    # train_transforms.transforms.append(Blur(k=3, sig=5))
    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )
    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(torch.cuda.is_available())
    print(device)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    net = models[args.model](quality=args.quality, pretrained=False)
    net = net.to(device)
    # if args.cuda and torch.cuda.device_count() > 1:
    #    net = CustomDataParallel(net)
    optimizer, aux_optimizer = configure_optimizers(net, args)
    #     lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=20)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400, 700], gamma=0.2)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    filename = "train" + str(args.quality) + ".csv"
    csv_logger = CSVLogger(
        fieldnames=['epoch', 'train_loss', 'train_bpp_loss', 'train_aux', 'test_loss', 'test_bpp_loss', 'test_aux'],
        filename=filename)
    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        #         for g in optimizer.param_groups:
        #             g['lr'] = 0.0001
        #         for g in aux_optimizer.param_groups:
        #             g['lr'] = 0.0001
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        start = time.time()
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_loss, train_bpp_loss, train_aux = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss, bpp_loss, aux = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step()

        row = {'epoch': str(epoch), 'train_loss': str(train_loss.item()), 'train_bpp_loss': str(train_bpp_loss.item()),
               'train_aux': str(train_aux.item()), 'test_loss': str(loss.item()), 'test_bpp_loss': str(bpp_loss.item()),
               'test_aux': str(aux.item())}
        csv_logger.writerow(row)  ###
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                str(args.quality)
            )
        print(f"Total TIme: {time.time() - start}")
    csv_logger.close()  ###


if __name__ == "__main__":
    main(sys.argv[1:])