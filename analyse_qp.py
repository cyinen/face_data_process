# -- coding: utf-8 --**
import threading
import os
import json
from traceback import print_tb
import fire
import queue
import csv

# followed https://github.com/slhck/ffmpeg-debug-qp to install ffmpeg_debug_qp

def analyse_qp(src_path, out_path):
    video_files = os.listdir(src_path)
    video_files.sort()
    for video_file in video_files:
        if video_file.endswith(".h264") or video_file.endswith(".enc"): # transfer to mp4
            video_prefix_name  =video_file.split('.')[0]
            
            input_video_path = os.path.join(src_path, video_file)
            output_mp4_path = os.path.join(out_path, video_prefix_name+".mp4")
            output_mp4_log_path = os.path.join(out_path, video_prefix_name+".log")

            transfer_command = (
                f'ffmpeg -y -f h264 '
                f' -i {input_video_path} '
                f' -vcodec copy {output_mp4_path}'
                f' 2> {output_mp4_log_path}'
            )
            os.system(transfer_command)

            output_json_path = os.path.join(out_path, video_prefix_name+".json")
            parse_command = (
                f'ffmpeg_debug_qp_parser -f {output_mp4_path} {output_json_path} -m -of json'
            )
            os.system(parse_command)

            qp_min = 99
            qp_max = 0
            frame_qp_min = 99
            frame_qp_max = 0
            static_dict = {}
            with open(output_json_path, "r") as fp_in:
                load_list = json.load(fp_in) # load json (as list)
                for frame_idx, frame in enumerate(load_list):
                    frame_aver_QP = frame['qpAvg']
                    static_dict[int(frame_idx)] = frame_aver_QP
                    frame_qp_min = frame_aver_QP if frame_aver_QP < frame_qp_min else frame_qp_min
                    frame_qp_max = frame_aver_QP if frame_aver_QP > frame_qp_max else frame_qp_max


                    frame_QP_values = frame['qpValues']
                    for mb_idx, mb in enumerate(frame_QP_values):
                        mb_qp = mb['qp']
                        qp_min = mb_qp if mb_qp < qp_min else qp_min
                        qp_max = mb_qp if mb_qp > qp_max else qp_max
            
            print(f'{video_prefix_name} : qp_min={qp_min},  qp_max={qp_max}')
            # video1_10 : qp_min=34,  qp_max=42
            # video1_100 : qp_min=25,  qp_max=42
            # video1_1024 : qp_min=16,  qp_max=34
            # video1_2048 : qp_min=16,  qp_max=34
            # video1_300 : qp_min=18,  qp_max=38

            print(f'{video_prefix_name} : frame_qp_min={frame_qp_min:.2f},  frame_qp_max={frame_qp_max:.2f}')
            # video1_10 : frame_qp_min=39.38,  frame_qp_max=42.00
            # video1_100 : frame_qp_min=30.63,  frame_qp_max=42.00
            # video1_1024 : frame_qp_min=16.96,  frame_qp_max=34.00
            # video1_2048 : frame_qp_min=16.00,  frame_qp_max=34.00
            # video1_300 : frame_qp_min=23.64,  frame_qp_max=38.00

            csv_file_path = os.path.join(out_path, video_prefix_name+".csv")
            with open(csv_file_path,"w") as csv_file:
                writer = csv.DictWriter(csv_file, static_dict.keys())
                writer.writeheader()
                writer.writerow(static_dict)


if __name__ == "__main__":
    fire.Fire()

# example
# python analyse_qp.py analyse_qp --src_path ./raw_video_file/ --out_path ./output/