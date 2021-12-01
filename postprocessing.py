import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from functools import partial
from tqdm import tqdm

import config


def readyuv420(filename, bitdepth, W, H, startframe, totalframe):
    #   startframe（ ）   （0-based），  totalframe
    Y = np.zeros((totalframe//config.frames_interval+1, H, W), np.uint8)

    bytes2num = partial(int.from_bytes, byteorder='little', signed=False)

    bytesPerPixel = math.ceil(bitdepth / 8)
    fp = open(filename, 'rb')

    j = 0
    seekPixels = startframe * H * W * 3 // 2
    fp.seek(bytesPerPixel * seekPixels)
    for i in range(totalframe+1):
        for m in range(H):
            for n in range(W):
                if bitdepth == 8:
                    pel = bytes2num(fp.read(1))
                    if i%config.frames_interval==0:
                        Y[j, m, n] = np.uint8(pel)
                elif bitdepth == 10:
                    pel = bytes2num(fp.read(2))
                    if i%config.frames_interval==0:
                        Y[j, m, n] = np.uint8(pel/4)
        if i % config.frames_interval == 0:
            j+=1

    if totalframe == 1:
        return Y[0]
    else:
        return Y

def show_img(b, img, img2):
    plt.ion()
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    if b:
        plt.subplot(132)
        plt.imshow(img2, cmap='gray')
    plt.show()
    plt.pause(1)
    # plt.pause(0.001)

def preprocessing(input_path, output_path, seq, array, poc):
    poc_idx = 0
    block_idx = 0
    f = open(input_path)
    lines = f.readlines()
    f.close()
    for line in lines:
        if len(line.split())!=0 and line.split()[0]=="POC":
            poc_idx+=1
            block_idx = 0
        if poc_idx == poc:
            info = line.strip().replace(" ", "").split("-left:")
            left = info[-1].split(":")
            info = info[0].split("-above:")
            above = info[-1].split(":")
            info = info[0].split("-")
            if info[0]=="INFO":
                calculate_np_array(output_path, seq+"_"+str(block_idx)+".npy", info, above, left, array)
                # calculate_np_array(output_path, seq+".npy", info, above, left, array)
                block_idx += 1

def calculate_np_array(base_path, seq, info, above, left, array):
    mode = info[2].split(":")[-1]
    wdt = int(info[5].split(":")[-1])
    hgt = int(info[6].split(":")[-1])
    if wdt!=32 or hgt!=32: return

    if min(wdt, hgt) <= 8:
        xa = min(wdt, hgt)
        ya = xa
    else:
        xa = wdt//2
        ya = hgt//2

# make picture
    h, w = array.shape
    picture = np.zeros((h+ya, w+xa), np.uint8)
    picture.fill(127)
    picture[ya:, xa:] = array

# make blocks
    above_block = np.zeros((ya, wdt * 2 + xa), np.uint8)
    left_block = np.zeros((hgt * 2, xa), np.uint8)
    y_block = np.zeros((hgt, wdt), np.uint8)

    x = int(info[3].split(":")[-1])+xa
    y = int(info[4].split(":")[-1])+ya

    y_block = picture[y:y + hgt, x:x + wdt]
    above_block = picture[y - ya:y, x - xa:x + wdt * 2]
    left_block = picture[y:y + hgt * 2, x - xa:x]

    block_sum = above_block.sum() + left_block.sum()
    block_mean_size = hgt*2*xa + (wdt*2+xa)*ya

    if left[0] == "0":
        none_x = int(left[1])+xa
        none_y = int(left[2])+ya
        if none_x>0:
            block_sum -= (left_block[none_y-y:, none_x-x:].sum())
            block_mean_size -= (left_block[none_y-y:, none_x-x:].shape[0] * left_block[none_y-y:, none_x-x:].shape[1])
            left_block[none_y-y:, none_x-x:].fill(255)
    if above[0] == "0":
        none_x = int(above[1]) + xa
        none_y = int(above[2]) + ya
        if none_y>0:
            block_sum -= (above_block[none_y - y:, none_x - x:].sum())
            block_mean_size -= (
                        above_block[none_y - y:, none_x - x:].shape[0] * above_block[none_y - y:, none_x - x:].shape[1])
            above_block[none_y - y:, none_x - x:].fill(255)

    block_mean = block_sum/block_mean_size
    y_block = y_block - block_mean
    above_block = above_block - block_mean
    left_block = left_block - block_mean

    output_path = os.path.join(base_path, str(hgt)+"x"+str(wdt))
    iter_save_np_array(y_block, above_block, left_block, output_path, seq)


def iter_save_np_array(y, above, left, output_path, seq):
    make_folder(output_path)
    save_y_path = os.path.join(output_path, "y", seq)
    save_above_path = os.path.join(output_path, "above", seq)
    save_left_path = os.path.join(output_path, "left", seq)
    np.save(save_y_path, y)
    np.save(save_above_path, above)
    np.save(save_left_path, left)

def seq_save_np_array(y, above, left, output_path, seq):
    make_folder(output_path)
    save_y_path = os.path.join(output_path, "y", seq)
    save_above_path = os.path.join(output_path, "above", seq)
    save_left_path = os.path.join(output_path, "left", seq)
    Y = np.load(save_y_path) if os.path.isfile(save_y_path) else []  # get data if exist
    np.save(save_y_path, np.append(Y, y))  # save the new
    Above = np.load(save_above_path) if os.path.isfile(save_above_path) else []  # get data if exist
    np.save(save_above_path, np.append(Above, above))  # save the new
    Left = np.load(save_left_path) if os.path.isfile(save_left_path) else []  # get data if exist
    np.save(save_left_path, np.append(Left, left))  # save the new

def make_folder(base_path):
    save_above_path = os.path.join(base_path, "above")
    if not os.path.exists(save_above_path):
        os.makedirs(save_above_path)
    save_left_path = os.path.join(base_path, "left")
    if not os.path.exists(save_left_path):
        os.makedirs(save_left_path)
    save_left_path = os.path.join(base_path, "y")
    if not os.path.exists(save_left_path):
        os.makedirs(save_left_path)

def main(i, base_path):
    qp_list = ['22', '27', '32', '37']

    for qp in qp_list:
        encoder_path = "output\\encoder\\"+i+"\\"+qp
        recon_path = "output\\recon\\"+i+"\\"+qp

        sequence_list = os.listdir(os.path.join(base_path, recon_path))
        for idx, sequence in enumerate(tqdm(sorted(sequence_list))):
            # convert_rgb2yuv420(base_path + "\\" + path_input, sequence)
            sequence = sequence[:-4]

            path_recon_file = os.path.join(base_path, recon_path, sequence+".yuv")
            path_log = os.path.join(base_path, encoder_path, sequence+".log")
            output_path = os.path.join(config.valid_numpy_path, qp)
            # output_path = os.path.join(config.train_numpy_path, qp)

            ######################## CLIC 2020 #########################
            # if sequence[:-5][-3:]=="png":
            #     img_path = os.path.join(base_path, "input", i, sequence[:-5])
            # else: img_path = os.path.join(base_path,"input",  i, sequence[:-5]+".png")
            # img = cv2.imread(img_path)
            # h, w, c = img.shape
            # Y = readyuv420(path_recon_file, 10,
            #                      w,
            #                      h, 0, 1)
            # show_img(False, y, None)
            # preprocessing(path_log, output_path, sequence[:-5], Y, 0)

            ##################### test sequence ########################
            total_frame = int(sequence.split("_")[-2])*2
            print(total_frame)
            size = sequence.split("_")[-3]
            print(sequence)
            h = int(size[:3])
            w = int(size[-3:])
            Y = readyuv420(path_recon_file, 10,
                           w,
                           h, 0, total_frame)
            for j in tqdm(range(len(Y))):
                preprocessing(path_log, output_path, sequence + "_"+str(j), Y[j], j*config.frames_interval)


if __name__ == "__main__":
    classes = ['Class_C', 'Class_D']
    # classes = ['train']
    base_path = config.bin_path
    for i in classes:
        main(i, base_path)



