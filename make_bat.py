import os
import sys
import argparse

import config


def main(file_path, cfg, path_input, path_bitstream, frame_rate, frame):
    qp_list = ['22', '27', '32', '37']
    lines = []

    path_bitstream = "output\\decoder\\" + path_bitstream

    for qp in qp_list:
        path_recon_file = path_input.split(".")[0]+"_qp"+qp+".yuv"
        path_log = path_input.split(".")[0]+"_qp"+qp+".log"
        enc_line = "EncoderApp.exe -c encoder_intra_vtm.cfg -c "+cfg+" -i "+path_input+" -b "+path_bitstream+" -q "+qp+" -fr "+frame_rate+" --ReconFile=output\\recon\\"+path_recon_file+" -fs 0 > output\\encoder\\"+path_log+"\n"
        dec_line = "DecoderApp.exe -b "+path_bitstream+" -o output\\decoder\\"+path_recon_file + " > output\\decoder\\"+path_log+"\n"
        lines.append(enc_line)
        lines.append("\n")
        #lines.append(dec_line)
        #lines.append("\n")

    with open(file_path, 'w') as f:
        f.writelines(lines)


if __name__ == "__main__":
    classes = ['Class_B', 'Class_C', 'Class_D']
    base_path = config.bin_path
    for i in classes:
        sequence_list = os.listdir(os.path.join(base_path, i))
        if not os.path.exists(base_path+"\\output\\encoder\\"+i):
            os.makedirs(base_path+"\\output\\encoder\\"+i)
        if not os.path.exists(base_path+"\\bin\\output\\decoder\\"+i):
            os.makedirs(base_path+"\\output\\decoder\\"+i)
        if not os.path.exists(base_path+"\\bin\\output\\recon\\"+i):
            os.makedirs(base_path+"\\output\\recon\\"+i)
        for j in sequence_list:
            path_input = i+"\\"+j
            sequence = j.split("_")[0]
            file_path = base_path +"\\"+ i +"_"+ sequence + ".bat"
            cfg = config.vtm_path + "\\cfg\\per-sequence\\" + sequence+".cfg"
            path_bitstream = i+"\\"+j.split(".")[0]+".bin"
            frame_rate = j.split("_")[2][:2]
            frame = str(int(frame_rate) * 2)

            main(file_path, cfg, path_input, path_bitstream, frame_rate, frame)
            print(file_path, cfg, path_input, path_bitstream, frame_rate, frame)

