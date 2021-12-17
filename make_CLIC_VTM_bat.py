import os
import sys
import argparse
import cv2

import config

output_folders = ['output', 'output_anchor']
output_path_number = 1

def convert_rgb2yuv420(input_path, sequence, size):
    savepath = "\\".join(input_path.split("\\")[:-1]) + "_yuv\\"+sequence.split(".")[0]+".yuv"
    print(savepath)

    convert = "ffmpeg -y -i "+input_path+" -s "+size+" -pix_fmt yuv420p "+savepath
    out = os.system(convert)
    print(out)

def main(sequence_list, i, base_path, output_folder):
    qp_list = ['22', '27', '32', '37']
    lines = []

    cfg = "train.cfg"
    file_path = base_path + "\\" + i + ".bat"

    for sequence in sequence_list:
        path_input = "input\\" + i + "\\" + sequence
        img = cv2.imread(base_path + "\\" + path_input)
        h, w, c = img.shape
        h+=(h%2)
        w+=(w%2)
        # convert_rgb2yuv420(base_path + "\\" + path_input, sequence, str(w)+"x"+str(h))

        sequence = sequence.split(".")[0]
        path_input_yuv = "input\\" + i + "_yuv\\"+sequence+".yuv"

        for qp in qp_list:
            decoder_path = output_folder+"\\decoder\\" + i + "\\" + qp + "\\"
            encoder_path = output_folder+"\\encoder\\"+i+"\\"+qp+"\\"
            recon_path = output_folder+"\\recon\\"+i+"\\"+qp+"\\"
            if not os.path.exists(os.path.join(base_path,decoder_path)):
                os.makedirs(os.path.join(base_path,decoder_path))
            if not os.path.exists(os.path.join(base_path,encoder_path)):
                os.makedirs(os.path.join(base_path,encoder_path))
            if not os.path.exists(os.path.join(base_path,recon_path)):
                os.makedirs(os.path.join(base_path,recon_path))
            path_bitstream = decoder_path + sequence + ".bin"

            path_recon_file = recon_path + sequence+"_qp"+qp+".yuv"
            path_predictor_file = recon_path + sequence+"_predictor_qp"+qp+".yuv"
            path_log = encoder_path+sequence+"_qp"+qp+".log"
            if output_path_number == 1:
                enc_line = "EncoderApp.exe -c encoder_intra_vtm.cfg -c " + cfg + " -i " + path_input_yuv + " -b " + path_bitstream + " -wdt " + str(
                    w) + " -hgt " + str(
                    h) + " -q " + qp + " --ReconFile=" + path_recon_file + " --PredictorFile=" + path_predictor_file +" -fs 0 > " + path_log + "\n"
            else:
                enc_line = "EncoderApp.exe -c encoder_intra_vtm.cfg -c "+cfg+" -i "+path_input_yuv+" -b "+path_bitstream+" -wdt "+str(w) + " -hgt "+str(h)+" -q "+qp+" --ReconFile="+path_recon_file+" -fs 0 > "+path_log+"\n"
            dec_line = "DecoderApp.exe -b "+path_bitstream+" -o "+path_recon_file + " > "+output_folder+"\\decoder\\"+qp+"\\"+path_log+"\n"
            lines.append(enc_line)
            lines.append("\n")
            #lines.append(dec_line)
            #lines.append("\n")

    with open(file_path, 'w') as f:
        f.writelines(lines)


if __name__ == "__main__":
    classes = ['train', 'valid', 'professional']
    base_path = config.bin_path
    for i in classes:
        sequence_list = os.listdir(os.path.join(base_path, "input\\" + i))
        main(sequence_list, i, base_path, output_folders[1])



