import argparse
import random
import shutil
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from torch.utils.tensorboard import SummaryWriter

import config

from train_cluster import load_cluster
from models.tapnn import TextureAdaptivePNN, block_size
from dataset.load_dataset import TAPNN
from util import AverageMeter, CSVLogger


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, help="Training dataset"
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
    parser.add_argument("--save", action="store_true", help="Save model to disk")
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument('--clusterk', type=int, default=0, help='cluster index (default: 0)')
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("-hgt", "--height", type=int, default=32, help="block height")
    parser.add_argument("-wdt", "--width", type=int, default=32, help="block width")
    args = parser.parse_args(argv)
    return args

def train_one_epoch(
        model, criterion, train_dataloader, optimizer, epoch
):
    model.train()
    device = next(model.parameters()).device

    loss = AverageMeter()
    for i, (a, l, y) in enumerate(train_dataloader):
        a = a.to(device)
        l = l.cuda()
        y = y.cuda()

        optimizer.zero_grad()
        out = model(a, l)
        out = out.view(y.shape)
        out_criterion = criterion(out, y)

        loss.update(out_criterion)
        out_criterion.backward()

        optimizer.step()
        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i * len(y)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion.item():.3f}'
            )
    del a
    del l
    del y
    # GPU memory delete
    torch.cuda.empty_cache()
    return loss.avg.item()

def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device
    loss = AverageMeter()
    with torch.no_grad():
        for (a, l, y) in test_dataloader:
            a = a.to(device)
            l = l.cuda()
            y = y.cuda()
            out = model(a, l)
            out = out.view(y.shape)
            out_criterion = criterion(out, y)
            loss.update(out_criterion)
    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f}\n"
    )
    del a
    del l
    del y
    # GPU memory delete
    torch.cuda.empty_cache()
    return loss.avg.item()


def save_checkpoint(state, is_best, q, h, w, k, filename="checkpoint\\"):
    torch.save(state, filename + h+"x"+w+"_"+ q  +"_" + k + ".pth.tar")
    if is_best:
        shutil.copyfile(filename +  h+"x"+w+"_"+ q  +"_" + k + ".pth.tar", filename + "best_loss_" + h+"x"+w+"_"+ q +"_" + k + ".pth.tar")


def main(argv):
    args = parse_args(argv)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    ############################################### load dataset ##############################################
    if args.dataset is None:
        args.dataset = config.train_numpy_path
    train_transforms = transforms.Compose(
         [transforms.ToTensor()]  # numpy이미지에서 torch이미지로 변경
    )
    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )
    path_model_cluster = os.path.join(config.cluster_checkpoint, str(args.height)+"x"+str(args.width) + "_" + str(args.quality) + '.pkl')
    km = load_cluster(path_model_cluster)

    train_dataset = TAPNN(os.path.join(args.dataset,  str(args.quality), str(args.height)+"x"+str(args.width)), args.height, args.width,
                          transform=train_transforms, km=km, k=args.clusterk)
    test_dataset = TAPNN(os.path.join(config.valid_numpy_path, str(args.quality), str(args.height)+"x"+str(args.width)), args.height, args.width,
                         transform=test_transforms, km=km, k=args.clusterk)
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=15)
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400, 700], gamma=0.2)
    criterion = nn.L1Loss()

    filename = "log_csv\\"+str(args.height)+"x"+str(args.width)+"_" + str(args.quality) +"_" + str(args.clusterk) + ".csv"
    csv_logger = CSVLogger(
        fieldnames=['epoch', 'train_loss', 'test_loss'],
        filename=filename)
    writer = SummaryWriter()

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        #         for g in optimizer.param_groups:
        #             g['lr'] = 0.0001
        #         for g in aux_optimizer.param_groups:
        #             g['lr'] = 0.0001
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        start = time.time()
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_loss = train_one_epoch(
            model,
            criterion,
            train_dataloader,
            optimizer,
            epoch,
        )
        loss = test_epoch(epoch, test_dataloader, model, criterion)
        lr_scheduler.step(loss)
        print(f"Train epoch {epoch}: "
                f'\tTrain Loss: {train_loss:.3f}'
              f'\tTest Loss: {loss:.3f}')

        writer.add_scalars("Loss_"+str(args.quality)+"_"+str(args.clusterk), {'train': train_loss,
                                    'valid': loss}, epoch)

        row = {'epoch': str(epoch), 'train_loss': str(train_loss), 'test_loss': str(loss)}
        csv_logger.writerow(row)  ###
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                str(args.quality), str(args.height), str(args.width), str(args.clusterk)
            )
        print(f"Total TIme: {time.time() - start}")
    csv_logger.close()  ###
    writer.flush()

if __name__ == "__main__":
    main(sys.argv[1:])