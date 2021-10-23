import os
import sys
import argparse
import cv2


def convert_rgb2yuv420(input_path, sequence):
    savepath = "\\".join(input_path.split("\\")[:-1]) + "_yuv\\"+sequence.split(".")[0]+".yuv"
    print(savepath)

    convert = "ffmpeg -y -i "+input_path+" -pix_fmt yuv420p "+savepath
    os.system(convert)

def main(sequence_list, i, base_path):
    qp_list = ['22', '27', '32', '37']
    lines = []

    cfg = "train.cfg"
    file_path = base_path + "\\" + i + ".bat"

    for sequence in sequence_list:
        path_input = i + "\\" + sequence
        img = cv2.imread(base_path + "\\" + path_input)
        h, w, c = img.shape
        #convert_rgb2yuv420(base_path + "\\" + path_input, sequence)
        sequence = sequence.split(".")[0]
        path_input = i + "_yuv\\"+sequence+".yuv"

        for qp in qp_list:
            decoder_path = "output\\decoder\\" + i + "\\" + qp + "\\"
            encoder_path = "output\\encoder\\"+i+"\\"+qp+"\\"
            recon_path = "output\\recon\\"+i+"\\"+qp+"\\"
            if not os.path.exists(os.path.join(base_path,decoder_path)):
                os.makedirs(os.path.join(base_path,decoder_path))
            if not os.path.exists(os.path.join(base_path,encoder_path)):
                os.makedirs(os.path.join(base_path,encoder_path))
            if not os.path.exists(os.path.join(base_path,recon_path)):
                os.makedirs(os.path.join(base_path,recon_path))
            path_bitstream = decoder_path + sequence + ".bin"

            path_recon_file = recon_path + sequence+"_qp"+qp+".yuv"
            path_log = encoder_path+sequence+"_qp"+qp+".log"
            enc_line = "EncoderApp.exe -c encoder_intra_vtm.cfg -c "+cfg+" -i "+path_input+" -b "+path_bitstream+" -wdt "+str(w) + " -hgt "+str(h)+" -q "+qp+" --ReconFile="+path_recon_file+" -fs 0 > "+path_log+"\n"
            dec_line = "DecoderApp.exe -b "+path_bitstream+" -o "+path_recon_file + " > output\\decoder\\"+qp+"\\"+path_log+"\n"
            lines.append(enc_line)
            lines.append("\n")
            #lines.append(dec_line)
            #lines.append("\n")

    with open(file_path, 'w') as f:
        f.writelines(lines)


if __name__ == "__main__":
    classes = ['train']
    base_path = "C:\\Users\\user\\Desktop\\VVCSoftware_VTM-VTM-9.0\\VTM\\bin"
    for i in classes:
        sequence_list = os.listdir(os.path.join(base_path, i))
        main(sequence_list, i, base_path)



