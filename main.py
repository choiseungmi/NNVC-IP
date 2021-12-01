import os
import time
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

import transforms as T
from inference import inference_user
from make_dataset.load_dataset import AIHUB, AIHUB_INFERENCE
from movinets import MoViNet
from movinets.config import _C
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import opts

model_list = {
    'movinetA0': _C.MODEL.MoViNetA0,
    'movinetA1': _C.MODEL.MoViNetA1,
    'movinetA2': _C.MODEL.MoViNetA2,
}


def train_iter(model, optimz, data_load, loss_val):
    samples = len(data_load.dataset)
    model.train()
    model.clean_activation_buffers()
    optimz.zero_grad()
    for i, (data, _, target) in enumerate(data_load):
        data = data.cuda()
        target = target.cuda()
        out = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(out, target)
        loss.backward()
        optimz.step()
        optimz.zero_grad()
        model.clean_activation_buffers()
        if i % 50 == 0:
            print('[' + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_load)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_val.append(loss.item())
        del data
        torch.cuda.empty_cache()


def evaluate(model, data_load, loss_val):
    model.eval()
    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0
    model.clean_activation_buffers()
    with torch.no_grad():
        for data, _, target in data_load:
            data = data.cuda()
            target = target.cuda()
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            tloss += loss.item()
            csamp += pred.eq(target).sum()
            model.clean_activation_buffers()
    aloss = tloss / samples
    loss_val.append(aloss)
    print('\nAverage test loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(100.0 * csamp / samples) + '%)\n')
    del data
    torch.cuda.empty_cache()


def train_iter_stream(model, optimz, data_load, loss_val, n_clips=opts.N_CLIPS, n_clip_frames=opts.N_CLIP_FRAMES):
    """
    In causal mode with stream buffer a single video is fed to the network
    using subclips of lenght n_clip_frames.
    n_clips*n_clip_frames should be equal to the total number of frames presents
    in the video.
    n_clips : number of clips that are used
    n_clip_frames : number of frame contained in each clip
    """
    # clean the buffer of activations
    samples = len(data_load.dataset)
    model.train()
    model.clean_activation_buffers()
    optimz.zero_grad()

    for i, (data, _, target) in enumerate(data_load):
        data = data.cuda()
        target = target.cuda()
        l_batch = 0
        # backward pass for each clip
        for j in range(n_clips):
            output = F.log_softmax(model(data[:, :, n_clip_frames * j:n_clip_frames * (j + 1)]), dim=1)
            _, pred = torch.max(output, dim=1)
            # nSamples = [1, 10, 15, 20, 20, 20, 20, 20, 20, 30]
            # normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
            # normedWeights = torch.FloatTensor(normedWeights).cuda()
            loss = F.nll_loss(output, target) / n_clips
            loss.backward()
        l_batch += loss.item() * n_clips
        optimz.step()
        optimz.zero_grad()

        # clean the buffer of activations
        model.clean_activation_buffers()
        if i % 50 == 0:
            print('[' + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_load)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(l_batch))
            loss_val.append(l_batch)
        del data
        torch.cuda.empty_cache()


def evaluate_stream(model, data_load, loss_val, n_clips=opts.N_CLIPS, n_clip_frames=opts.N_CLIP_FRAMES):
    model.eval()
    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0
    with torch.no_grad():
        for data, _, target in data_load:
            data = data.cuda()
            target = target.cuda()
            model.clean_activation_buffers()
            for j in range(n_clips):
                output = F.log_softmax(model(data[:, :, (n_clip_frames) * (j):(n_clip_frames) * (j + 1)]), dim=1)
                loss = F.nll_loss(output, target)
            _, pred = torch.max(output, dim=1)
            tloss += loss.item()
            csamp += pred.eq(target).sum()

    aloss = tloss / len(data_load)
    loss_val.append(aloss)
    acc = 100.0 * csamp / samples
    print('\nAverage test loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(acc) + '%)\n')
    del data
    # GPU memory delete
    torch.cuda.empty_cache()
    return acc, 0, 0


def evaluate_none(model, data_load, loss_val, n_clips=opts.N_CLIPS, n_clip_frames=opts.N_CLIP_FRAMES):
    model.eval()
    samples = len(data_load.dataset)
    labels = 0
    nones = 0
    csamp_label = 0
    csamp_none = 0
    tloss = 0
    with torch.no_grad():
        for data, _, target in tqdm(data_load):
            data = data.cuda()
            target = target.cuda()
            model.clean_activation_buffers()
            for j in range(n_clips):
                output = F.log_softmax(model(data[:, :, n_clip_frames * j:n_clip_frames * (j + 1)]), dim=1)
                loss = F.nll_loss(output, target)
                # a = [[0.7 for i in range(10)] for k in range(len(data))]
                # b = torch.log(torch.FloatTensor(a)).cuda()
                # output = output.ge(b)
            _, pred = torch.max(output, dim=1)
            pred = torch.IntTensor([3 if value == False else pred[i] for i, value in enumerate(_)]).cuda()
            pred = pred.clamp(max=3)
            a = [2 for i in range(len(target))]
            b = torch.FloatTensor(a).cuda()
            tloss += loss.item()
            compare = target.le(b)
            labels += compare.sum()
            nones += (~compare).sum()
            compare_label = pred.eq(target)
            compare_none = pred.gt(b)
            csamp_label += (compare & compare_label).sum()
            csamp_none += (~compare & compare_none).sum()

    aloss = tloss / len(data_load)
    loss_val.append(aloss)
    csamp = csamp_label + csamp_none
    acc = 100.0 * csamp / samples
    print('\nAverage test loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(acc) + '%)\n')
    acc_label = 100.0 * csamp_label / labels
    acc_none = 100.0 * csamp_none / nones
    print('Label Accuracy: ' + '{:5}'.format(csamp_label) + '/' +
          '{:5}'.format(labels) + ' (' +
          '{:4.2f}'.format(acc_label) + '%)\n' +
          '  None Accuracy:' + '{:5}'.format(csamp_none) + '/' +
          '{:5}'.format(nones) + ' (' +
          '{:4.2f}'.format(acc_none) + '%)\n')
    del data
    # GPU memory delete
    torch.cuda.empty_cache()
    return acc, acc_label, acc_none


def train():
    transform = transforms.Compose([
        T.ToFloatTensorInZeroOne(),
        T.Resize((200, 200)),
        T.RandomHorizontalFlip(),
        # T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        # T.RandomVerticalFlip(),
        T.RandomPerspective(p=0.8),
        T.RandomCrop((172, 172))])
    transform_test = transforms.Compose([
        T.ToFloatTensorInZeroOne(),
        T.Resize((200, 200)),
        # T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        T.CenterCrop((172, 172))])

    train_data = AIHUB(opts.TRAIN_DATA_PATH, transform=transform)
    valid_data = AIHUB(opts.VALID_DATA_PATH, transform=transform_test)

    train_loader = DataLoader(train_data, batch_size=opts.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=opts.BATCH_SIZE, shuffle=False)

    N_EPOCHS = 50

    # MODEL DEFINE
    # MoviNetA0, ~ A5
    model = MoViNet(model_list[opts.MODEL], causal=True, pretrained=opts.PRETRAIN, num_classes=opts.CLASSES,
                    conv_type="2plus1d")
    model.classifier[3] = torch.nn.Conv3d(2048, opts.CLASSES, (1, 1, 1))
    model = model.cuda()

    start_time = time.time()
    trloss_val, tsloss_val = [], []
    optimz = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = MultiStepLR(optimz, milestones=[30, 50], gamma=0.1)

    # model = torch.load('model.pt')
    #######################  load_model  ###########################
    # checkpoint = torch.load("best_model.pt")
    # model.load_state_dict(checkpoint['model_state_dict'])
    # print(model.classifier[3])
    # optimz.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # scheduler.load_state_dict(checkpoint["scheduler"])
    ################################################################
    writer = SummaryWriter()
    best_acc = 0

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {pytorch_total_params}")

    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)
        train_iter_stream(model, optimz, train_loader, trloss_val)
        acc, acc_label, acc_none = evaluate_stream(model, valid_loader, tsloss_val)
        scheduler.step()
        writer.add_scalars("Loss", {'train': np.mean(np.array(trloss_val)),
                                    'valid': np.mean(np.array(tsloss_val))}, epoch)
        writer.add_scalars("Accuracy", {'Total': acc, 'Label': acc_label, 'None': acc_none}, epoch)
        if acc > best_acc:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimz.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, 'best_model.pt')
            best_acc = acc
            torch.save(model, 'model.pt')
    writer.flush()
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')


def test_model():
    # TEST LOADER
    #########################################################################################################
    model = MoViNet(model_list[opts.MODEL], causal=True, pretrained=opts.PRETRAIN, num_classes=opts.CLASSES,
                    conv_type="2plus1d")
    checkpoint = torch.load("best_model_A2_user_0.0005.pt")
    model.classifier[3] = torch.nn.Conv3d(2048, opts.CLASSES, (1, 1, 1))
    model = model.cuda()
    model.load_state_dict(checkpoint['model_state_dict'])

    transform_test = transforms.Compose([
        T.ToFloatTensorInZeroOne(),
        T.Resize((200, 200)),
        # T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        T.CenterCrop((172, 172))])
    test_data = AIHUB(opts.TEST_DATA_PATH, transform=transform_test)
    test_loader = DataLoader(test_data, batch_size=opts.BATCH_SIZE, shuffle=False)
    #######################################################################################################

    start_time = time.time()

    tsloss_val = []
    evaluate_stream(model, test_loader, tsloss_val)

    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')


if __name__ == '__main__':
    model_name = "modelA0"

    # train()
    test_model()

    # INFERENCE
    # #######################################################################################################
    # model = MoViNet(model_list[MODEL], causal=True, pretrained=PRETRAIN, num_classes=CLASSES, conv_type="2plus1d")
    # # checkpoint = torch.load("best_model_A2_user_0.0005.pt")
    # checkpoint = torch.load("best_model.pt")
    # model.classifier[3] = torch.nn.Conv3d(2048, CLASSES, (1, 1, 1))
    # model = model.cuda()
    # model.load_state_dict(checkpoint['model_state_dict'])
    #
    # path = TEST_DATA_PATH
    # for i in sorted(os.listdir(path)):
    #     inference_user(model, os.path.join(path, i), i)
    #     # inference_none(model, os.path.join(path, i), i)
    # #######################################################################################################