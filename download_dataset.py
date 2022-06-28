from cmath import e
import os
import fire
import time
import shutil
import platform
import youtube_dl
import multiprocessing
import traceback
import glob
import cv2
from dataset_prepare import print_run_time

if "windows" in platform.system().lower():
    TMP_ROOT = "C:/tmp/"
else:
    TMP_ROOT = "/tmp/"

def download_youtube_video(processes_id, videoid, output_video_dir_path, attempt_times=3):

    ydl_opts = {
            'format': 'bestvideo/best',
            'outtmpl': output_video_dir_path + '/%(id)s.%(ext)s',
            # 'merge_output_format': 'mp4',
            # 'format': '133',
            'ignore-errors': True,
            'external_downloader': 'aria2c',
            'external_downloader_args': [f'--log=/tmp/aria2c_{videoid}.log ', f'--max-concurrent-downloads={10}', f'--max-connection-per-server={4}'],
            'quiet': True # Do not print messages to stdout.
        }

    attempt_count = 0
    while(attempt_count < attempt_times):
        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.cache.remove()
                ydl.download([f'https://www.youtube.com/watch?v={videoid}'])

        except Exception as ex:
            print(f"{processes_id}({videoid}) attempt_count={attempt_count}: The following exception occurs: {type(ex)} {ex}", flush=True)
            # print(traceback.format_exc())
            attempt_count += 1
            time.sleep(10)
            continue
        finally:
            if(processes_id % 100 ==0):
                print(f'processes {processes_id}({videoid}) is done!', flush=True)
            return 0
    print(f"ERROR!!!!!!!!!!!!!!1  {processes_id}({videoid}) attempt_count={attempt_count}", flush=True)

def download_youtube_video_yt_dlp(processes_id, class_type, videoid, height, output_video_dir_path, attempt_times=3):
    try:
        os.system(
                f' yt-dlp --quiet '
                f' -o "{output_video_dir_path}/{height}p/{class_type}_{videoid}.mp4" '
                f' "https://www.youtube.com/watch?v={videoid}" '
                f' -S "height:{height}" '
                f' -f "bv[ext=mp4]" '
            )
    except Exception as ex:
        print(f"{processes_id}({class_type}_{videoid}): The following exception occurs: {type(ex)} {ex}", flush=True)
        print(traceback.format_exc())
    finally:
        if(processes_id % 100 ==0):
            print(f'processes {processes_id}({videoid}) is done!', flush=True)
        return 0

def download_youtube_video_multi_process(VFHQ_meta_info_root, output_video_dir_path, max_process=20):
    begin_time = time.time()

    processes_num = 0
    pool = multiprocessing.Pool(processes=max_process)
    
    meta_info_file_list = os.listdir(VFHQ_meta_info_root)
    meta_info_file_list.sort()

    fp_out = open("./address.txt", 'w')
    down_load_list = []
    for meta_info_file in meta_info_file_list:
        _, videoid, pid, clip_idx, frame_rlt = meta_info_file.split('+')
        if videoid in down_load_list:
            continue
        if(os.path.exists(f'{output_video_dir_path}/{videoid}.mp4') 
                or os.path.exists(f'{output_video_dir_path}/{videoid}.webm')):
            print(f'{videoid} already exits, skip it.')
            continue
        # pool.apply_async(download_youtube_video,
        #                 args=(processes_num, videoid, output_video_dir_path, ))
        processes_num += 1
        down_load_list.append(videoid)
        # time.sleep(1)
        fp_out.write(f'https://www.youtube.com/watch?v={videoid}\n')

    print(f'wait for {processes_num} processec done.')    
    pool.close()
    pool.join()
    print(f'There are {processes_num} videos to be download.')
    end_time = time.time()
    print_run_time(round(end_time-begin_time))

def download_youtube_video_from_addr_multi_process(output_video_dir_path, max_process=20):
    os.makedirs(f"{output_video_dir_path}/1080p", exist_ok=True)
    os.makedirs(f"{output_video_dir_path}/360p", exist_ok=True)
    begin_time = time.time()

    processes_num = 0
    pool = multiprocessing.Pool(processes=max_process)

    address_file_fp_in = open("./address_628.txt", 'r')
    down_load_list = address_file_fp_in.read().splitlines()
    for addr in down_load_list:
        class_type = addr.split(",")[0]
        addr = addr.split(",")[1]
        videoid = addr.split('?v=')[-1]
        pool.apply_async(download_youtube_video_yt_dlp,
                        args=(processes_num, class_type, videoid,1080, output_video_dir_path, ))
        pool.apply_async(download_youtube_video_yt_dlp,
                        args=(processes_num, class_type, videoid, 360, output_video_dir_path, ))
        processes_num += 1

    print(f'wait for {processes_num} processec done.')    
    pool.close()
    pool.join()
    print(f'There are {processes_num} videos to be download.')
    end_time = time.time()
    print_run_time(round(end_time-begin_time))

def get_crop_face(processes_id, raw_video_path, VFHQ_meta_info_root, meta_info_file_list, output_video_dir_path):
    try:
        for idx, meta_info_file in enumerate(meta_info_file_list):
            clip_name = meta_info_file.split(".")[0] # rm .mp4 and .webm
            _, videoid, pid, clip_idx, frame_rlt = clip_name.split('+')

            meta_info_file_path = os.path.join(VFHQ_meta_info_root, meta_info_file)
            clip_meta_file_fp = open(meta_info_file_path, 'r')
            for line in clip_meta_file_fp:
                if line.startswith('FPS'):
                    clip_fps = float(line.strip().split(' ')[-1])
                # get the coordinates of face
                if line.startswith('CROP'):
                    clip_crop_bbox = line.strip().split(' ')[-4:]
                    x0 = int(clip_crop_bbox[0])
                    y0 = int(clip_crop_bbox[1])
                    x1 = int(clip_crop_bbox[2])
                    y1 = int(clip_crop_bbox[3])
            
            clip_idx = int(clip_idx.split('C')[1])
            frame_start, frame_end = frame_rlt.replace('F', '').split('-')
            frame_start, frame_end = int(frame_start) + 1, int(frame_end) - 1
            start_t = round(frame_start / float(clip_fps), 5)
            end_t = round(frame_end / float(clip_fps), 5)

            save_clip_root = os.path.join(TMP_ROOT, clip_name)
            os.makedirs(save_clip_root, exist_ok=True)

            save_cropped_face_clip_root = os.path.join(output_video_dir_path, f'{videoid}-{clip_idx}')
            os.makedirs(save_cropped_face_clip_root, exist_ok=True)
            
            os.system(
                f' ffmpeg -loglevel error '
                f' -i {raw_video_path} '
                f' -an -vf "select=between(n\,{frame_start}\,{frame_end}),setpts=PTS-STARTPTS"  '
                f'-qscale:v 1 -qmin 1 -qmax 1 -vsync 0 '
                f' {save_clip_root}/%08d.png'
            )

            # crop the HQ frames
            hq_frame_list = sorted(glob.glob(os.path.join(save_clip_root, '*')))
            for frame_path in hq_frame_list:
                basename = os.path.splitext(os.path.basename(frame_path))[0]
                frame = cv2.imread(frame_path)
                cropped_face = frame[y0:y1, x0:x1]
                save_cropped_face_path = os.path.join(save_cropped_face_clip_root, f'{basename}.png')
                cv2.imwrite(save_cropped_face_path, cropped_face, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            
            shutil.rmtree(save_clip_root)
            # ffmpeg_tovideo_command = (
            #     f' ffmpeg -loglevel error '
            #     f' -i {save_cropped_face_clip_root}/%08d.png '
            #     f' -c:v libx264 -crf 0 '
            #     f' {output_video_dir_path}/{clip_name}.mp4'
            # )
            # os.system(ffmpeg_tovideo_command)
    except Exception as ex:
        print("process {processes_id}: The following exception occurs", type(ex), ": ", ex)
        print(traceback.format_exc())
    finally:
        if(processes_id % 100 ==0):
            print(f'processes {processes_id}({videoid}) is done!', flush=True)
        return 0

def get_crop_face_multi_process(raw_video_dir_path, VFHQ_meta_info_root, output_video_dir_path, max_process=1):
    begin_time = time.time()

    file_info_dict = {}
    
    meta_info_file_list = os.listdir(VFHQ_meta_info_root)
    meta_info_file_list.sort()
    for meta_info_file in meta_info_file_list:
        _, videoid, pid, clip_idx, frame_rlt = meta_info_file.split('+')
        if videoid not in file_info_dict:
            file_info_dict[videoid] = []
        file_info_dict[videoid].append(meta_info_file)

    processes_num = 0
    pool = multiprocessing.Pool(processes=max_process)
    
    raw_video_list = os.listdir(raw_video_dir_path)
    raw_video_list.sort()
    for raw_video_name in raw_video_list:
        if(raw_video_name.endswith(".part")):
            continue # the video was not download successfully, skip it!
        raw_video_path = os.path.join(raw_video_dir_path, raw_video_name)
        video_id = raw_video_name.split('.')[0]
        pool.apply_async(get_crop_face,
                        args=(processes_num, raw_video_path, VFHQ_meta_info_root,
                        file_info_dict[video_id], output_video_dir_path,))
        processes_num += 1
    print(f'wait for {processes_num} processec done.')    
    pool.close()
    pool.join()
    end_time = time.time()
    print_run_time(round(end_time-begin_time))


if __name__ == "__main__":
    fire.Fire()
    download_youtube_video_from_addr_multi_process("tmp")
    # download_youtube_video_from_addr_multi_process(
    #     VFHQ_meta_info_root='D:/huiguohe/data/VFHQ/VFHQ_meta_info/',
    #     output_video_dir_path='D:/huiguohe/data/VFHQ/raw_video/'
    # )

    # get_crop_face_multi_process(
    #     raw_video_dir_path='D:/huiguohe/data/VFHQ/raw_video/',
    #     VFHQ_meta_info_root='D:/huiguohe/data/VFHQ/VFHQ_meta_info/',
    #     output_video_dir_path='D:/huiguohe/data/VFHQ/crop_face/',
    #     max_process=10
    # )
    
    # download_youtube_video(processes_id=0, videoid='BU3KBl8IxIw',output_video_dir_path= f'D:/huiguohe/data/VFHQ/raw_video/')
# example

# python ./download_dataset.py download_youtube_video_multi_process --VFHQ_meta_info_root /data04/huiguohe/VFHQ/VFHQ_meta_info/VFHQ-Test/ --output_video_dir_path /data04/huiguohe/VFHQ/raw_video/
# python ./download_dataset.py download_youtube_video_multi_process --VFHQ_meta_info_root D:\huiguohe\data\VFHQ\VFHQ_meta_info\VFHQ-Train\ --output_video_dir_path D:\huiguohe\data\VFHQ\raw_video\
# python ./download_dataset.py download_youtube_video_multi_process --VFHQ_meta_info_root D:\huiguohe\data\VFHQ\VFHQ_meta_info\VFHQ-Test\ --output_video_dir_path D:\huiguohe\data\VFHQ\raw_video\

# python ./download_dataset.py download_youtube_video_from_addr_multi_process --output_video_dir_path /data04/huiguohe/VFHQ/raw_video/

# python ./download_dataset.py get_crop_face --raw_video /data04/huiguohe/VFHQ/raw_video_test/ --VFHQ_meta_info_root /data04/huiguohe/VFHQ/VFHQ_meta_info/VFHQ-Test/ --output_video_dir_path ./tmp/
