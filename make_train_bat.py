import os
import sys
import argparse

def main(sequence_list, i, base_path):
    qp_list = ['22', '27', '32', '37']
    lines = []

    cfg = base_path + "\\bin\\train.cfg"
    file_path = base_path + "\\bin\\" + i + ".bat"

    for sequence in sequence_list:
        path_input = i + "\\" + sequence
        path_bitstream = "output\\decoder\\" + i + "\\" + sequence.split(".")[0] + ".bin"

        for qp in qp_list:
            path_recon_file = path_input.split(".")[0]+"_qp"+qp+".yuv"
            path_log = path_input.split(".")[0]+"_qp"+qp+".log"
            enc_line = "EncoderApp.exe -c encoder_intra_vtm.cfg -c "+cfg+" -i "+path_input+" -b "+path_bitstream+" -q "+qp+" --ReconFile=output\\recon\\"+path_recon_file+" -fs 0 > output\\encoder\\"+path_log+"\n"
            dec_line = "DecoderApp.exe -b "+path_bitstream+" -o output\\decoder\\"+path_recon_file + " > output\\decoder\\"+path_log+"\n"
            lines.append(enc_line)
            lines.append("\n")
            #lines.append(dec_line)
            #lines.append("\n")

        with open(file_path, 'w') as f:
            f.writelines(lines)


if __name__ == "__main__":
    classes = ['train']
    base_path = "C:\\Users\\user\\Desktop\\VVCSoftware_VTM-VTM-9.0\\VTM"
    for i in classes:
        sequence_list = os.listdir(os.path.join(base_path,"bin", i))
        if not os.path.exists(base_path+"\\bin\\output\\encoder\\"+i):
            os.makedirs(base_path+"\\bin\\output\\encoder\\"+i)
        if not os.path.exists(base_path+"\\bin\\output\\decoder\\"+i):
            os.makedirs(base_path+"\\bin\\output\\decoder\\"+i)
        if not os.path.exists(base_path+"\\bin\\output\\recon\\"+i):
            os.makedirs(base_path+"\\bin\\output\\recon\\"+i)
        main(sequence_list, i, base_path)



