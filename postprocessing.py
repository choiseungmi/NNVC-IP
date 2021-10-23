import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_yuv(input, wdt, hgt):
    offset = int(wdt*hgt*3/2)
    frame = np.fromfile(input, dtype='uint8', offset=offset)

    Y = frame[0:wdt*hgt].reshape(hgt, wdt)
    plt.imshow(Y, cmap='gray')
    plt.show()

def convert_10bit_to_8bit(input_path, output_path):
    convert = "ffmpeg -f -i "+input_path+" -c:v libx265 -vf format=yuv420p -c:a copy "+output_path
    os.system(convert)

def main(i, base_path):
    qp_list = ['22', '27', '32', '37']

    for qp in qp_list:
        encoder_path = "output\\encoder\\"+i+"\\"+qp
        recon_path = "output\\recon\\"+i+"\\"+qp

        sequence_list = os.listdir(os.path.join(base_path, encoder_path))
        for sequence in sequence_list:
            # convert_rgb2yuv420(base_path + "\\" + path_input, sequence)

            path_recon_file = os.path.join(base_path, recon_path, sequence.split(".")[0]+".yuv")
            path_log = os.path.join(base_path, encoder_path, sequence)
            output_path = os.path.join(base_path, recon_path, sequence.split(".")[0]+"_8bit.yuv")
            convert_10bit_to_8bit(path_recon_file, output_path)
            read_yuv(output_path)


if __name__ == "__main__":
    classes = ['train']
    base_path = "C:\\Users\\user\\Desktop\\VVCSoftware_VTM-VTM-9.0\\VTM\\bin"
    for i in classes:
        main(i, base_path)



