import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from functools import partial

def read_yuv(input, wdt, hgt):
    offset = int(wdt*hgt*3/2)
    frame = np.fromfile(input, dtype='uint8', offset=offset)

    Y = frame[0:wdt*hgt].reshape(hgt, wdt)
    plt.imshow(Y, cmap='gray')
    plt.show()

def convert_10bit_to_8bit(input_path, output_path):
    convert = "ffmpeg -i "+input_path+" -c:v libx265 -vf format=yuv420p -c:a copy "+output_path
    os.system(convert)


def readyuv420(filename, bitdepth, W, H, startframe, totalframe, show=False):
    #   startframe（ ）   （0-based），  totalframe

    Y = np.zeros((totalframe, H, W), np.uint8)

    plt.ion()

    bytes2num = partial(int.from_bytes, byteorder='little', signed=False)

    bytesPerPixel = math.ceil(bitdepth / 8)
    fp = open(filename, 'rb')

    for i in range(totalframe):
        seekPixels = startframe * H * W * 3 // 2
        fp.seek(bytesPerPixel * seekPixels)
        for m in range(H):
            for n in range(W):
                if bitdepth == 8:
                    pel = bytes2num(fp.read(1))
                    Y[i, m, n] = np.uint8(pel)
                elif bitdepth == 10:
                    pel = bytes2num(fp.read(2))
                    Y[i, m, n] = np.uint8(pel/4)

        if show:
            print(i)
            plt.subplot(131)
            plt.imshow(Y[i, :, :], cmap='gray')
            plt.show()
            plt.pause(1)
            # plt.pause(0.001)

    if totalframe == 1:
        return Y[0]
    else:
        return Y


def main(i, base_path):
    qp_list = ['22', '27', '32', '37']

    for qp in qp_list:
        encoder_path = "output\\encoder\\"+i+"\\"+qp
        recon_path = "output\\recon\\"+i+"\\"+qp

        sequence_list = os.listdir(os.path.join(base_path, recon_path))
        for sequence in sorted(sequence_list):
            # convert_rgb2yuv420(base_path + "\\" + path_input, sequence)
            sequence = sequence[:-4]

            path_recon_file = os.path.join(base_path, recon_path, sequence+".yuv")
            path_log = os.path.join(base_path, encoder_path, sequence+".log")
            output_path = os.path.join(base_path, recon_path, sequence+"_8bit.yuv")
            # convert_10bit_to_8bit(path_recon_file, output_path)
            input_path = os.path.join(base_path, "train", sequence[:-5]+".png")
            img = cv2.imread(input_path)
            h, w, c = img.shape
            #read_yuv(output_path, w, h)
            y = readyuv420(path_recon_file, 10,
                                 w,
                                 h, 0, 1, True)
            print(y.shape)


if __name__ == "__main__":
    classes = ['train']
    base_path = "C:\\Users\\user\\Desktop\\VVCSoftware_VTM-VTM-9.0\\VTM\\bin"
    for i in classes:
        main(i, base_path)



