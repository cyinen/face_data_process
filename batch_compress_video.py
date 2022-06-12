# -- coding: utf-8 --**
import os
import fire
import time
import cv2
import traceback
import multiprocessing
import shutil
import numpy as np 

# Probability distribution for the cropped_video(.png)
# pick_bitrate_list = [50,     100,   200,    300,    400,    500,    600,    700,    800,    900]
# pick_p_list =       [6,      10,     20,     30,     20,     10,      8,      6,      3,      1]

# Probability distribution for the raw_video(.mp4)
pick_bitrate_list = [100,   200,    300,    400,    500,    600,    700,    800,    900]
pick_p_list =       [1,     3,      10,     20,     30,     20,     10,     3,      1]

sum_value = sum(pick_p_list) # normalize
for idx, p in enumerate(pick_p_list):
    pick_p_list[idx] = p / sum_value

def print_run_time(run_time):
    hour = run_time//3600
    minute = (run_time-3600*hour)//60
    second = run_time-3600*hour-60*minute
    print (f'The program run time ：{hour} hours {minute} minutes {second} seconds.')

def check_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def rm_video_dir(path):
    shutil.rmtree(path)

def get_bitrate(scale): # implement bit-rate generator here
    base_bitrate = np.random.choice(pick_bitrate_list, p=pick_p_list)
    return int(base_bitrate/scale) # /scale/scale tends to make too much video bitrate to small, so just /scale once

def update_bitrate(bitrate, scale):
    raw_bitrate = bitrate * scale
    for idx, br in enumerate(pick_bitrate_list):
        if(raw_bitrate == br and idx < len(pick_bitrate_list)-1):
            return pick_bitrate_list[idx+1] / scale
    return bitrate * 1.5

def get_frame_number(log_file):
    with open(log_file, 'r') as fp:
        one_line = fp.readline()
        while one_line:
            if(one_line[:-1].endswith("frames")):
                frame_num = int(one_line.split(' ')[-2])
                return frame_num
            one_line = fp.readline()
    return 0

def compress_video_process(processes_id, video_name, gt_videos_dir_path, output_dir, tmp_dir_root, scale_list):
    try:
        gt_video_path = os.path.join(gt_videos_dir_path, video_name)
        if(os.path.isdir(gt_video_path)):
            tmp_output_dir = os.path.join(tmp_dir_root, video_name)
            check_make_dir(tmp_output_dir)

            gt_imgs_list = os.listdir(gt_video_path)
            gt_imgs_list.sort()
            gt_len = len(gt_imgs_list)
            start_number = int(gt_imgs_list[0][-8:-4])
            height, width, channels = cv2.imread(os.path.join(gt_video_path, gt_imgs_list[0])).shape
        else:
            video_name = video_name[:-4]
            tmp_output_dir = os.path.join(tmp_dir_root, video_name)
            check_make_dir(tmp_output_dir)
            cap = cv2.VideoCapture(gt_video_path)
            gt_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            # height, width = 1080, 1920 # fix for deepfake dataset

        for scale in scale_list:
            bitrate = get_bitrate(scale)
            
            while(True):
                if(os.path.isdir(gt_video_path)):
                    # transfer *.png to .yuv and down_scale
                    gt_yuv_file = f' {os.path.join(tmp_output_dir, video_name)}_{width//scale}x{height//scale}.yuv'
                    trans_to_yuv_command = (
                        f'ffmpeg -y -loglevel error -hide_banner -nostats '
                        f' -start_number {start_number} '
                        f' -i {os.path.join(gt_video_path, video_name)}_%04d.png '
                        f' -vf scale={width//scale}:{height//scale} '
                        f' -pix_fmt yuv420p '
                        f' {gt_yuv_file}'
                    )
                    os.system(trans_to_yuv_command)
                else:
                    assert gt_video_path.endswith(".mp4")
                    gt_yuv_file = os.path.join(tmp_output_dir, f'{video_name}_{width//scale}x{height//scale}.yuv')

                    # transfer *.mp4 to .yuv and down_scale
                    trans_to_yuv_command = (
                        f'ffmpeg -y -loglevel error -hide_banner -nostats '
                        f' -i {gt_video_path}'
                        f' -vf scale={width//scale}:{height//scale}'
                        f' -pix_fmt yuv420p '
                        f' {gt_yuv_file}'
                    )
                    os.system(trans_to_yuv_command)

                # encode yuv video with teams codec MleTest.exe
                output_mp4_file = os.path.join(output_dir, f'down_x{scale}', f'{video_name}_{bitrate}.mp4')
                output_log = os.path.join(tmp_output_dir, f'{video_name}_down_x{scale}.txt')
                teams_codec_encode_command = (
                    f'{os.getcwd()}/release/MleTest.exe '
                    f' -h {height//scale} -w {width//scale} '
                    f' -rate {bitrate} '
                    f' -i {gt_yuv_file} '
                    f' -o {output_mp4_file} '
                    f' > {output_log} '
                )
                status = os.system(teams_codec_encode_command)

                # get number of frames
                num_frame = get_frame_number(output_log)
                if(num_frame == gt_len):
                    break
                else:
                    print(f'processes {processes_id}({video_name}_down_x{scale}): bit-rate({bitrate}) too low, enlarge it')
                    os.remove(output_mp4_file)
                    bitrate = update_bitrate(bitrate,scale)

        rm_video_dir(tmp_output_dir)

    except Exception as ex:
        print("The following exception occurs", type(ex), ": ", ex)
        print(traceback.format_exc())
    finally:
        if(processes_id % 100 ==0):
            print(f'processes {processes_id}({video_name}) is done!')
        return 0

def compress_video_multi_process(gt_videos_dir_path, output_dir, scale_list=[1,2,4], max_process=48):
    begin_time = time.time()
    tmp_dir_root = "./tmp"
    check_make_dir(tmp_dir_root)

    for scale in scale_list:
        down_dir_path = os.path.join(output_dir, f'down_x{scale}')
        check_make_dir(down_dir_path)

    processes_num = 0
    pool = multiprocessing.Pool(processes=max_process)
    gt_video_list = os.listdir(gt_videos_dir_path)
    gt_video_list.sort()
    for video_name in gt_video_list:
        pool.apply_async(compress_video_process,
                    args=(processes_num, video_name, gt_videos_dir_path, output_dir, tmp_dir_root, scale_list))
        processes_num += 1
    
    print(f'{processes_num} to be processed.')
    pool.close()
    pool.join()
    end_time = time.time()
    print_run_time(round(end_time-begin_time))

def remove_zero_video(video_dir_path):
    video_file_list = os.listdir(video_dir_path)
    video_file_list.sort()
    for video_file in video_file_list:
        video_path = os.path.join(video_dir_path, video_file)
        if(video_file.endswith('.mp4')):
            video_name, bitrate, scale = video_file[:-4].split('_')        
            if(os.path.getsize(video_path) == 0):
                os.remove(video_path)

if __name__ == "__main__":
    fire.Fire()
    # compress_video_multi_process(gt_videos_dir_path="./dataset/raw_video",
    #                 output_dir="./downsample_video/", scale_list=[1], max_process=12)

    # example:
    # python ./batch_compress_video.py remove_zero_video --video_dir_path

