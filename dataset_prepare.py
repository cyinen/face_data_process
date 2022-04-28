# -- coding: utf-8 --**
import threading
import os
import json
import fire
import queue
from PIL import Image, ImageDraw
from matplotlib.pyplot import close
import numpy as np
import cv2
import traceback
import time
import csv
import random
import multiprocessing
import math
import copy

import face_recognition
from deepface.detectors import FaceDetector

file_video_queue = queue.Queue() # only used for multi-thread
face_info_queue = queue.Queue()
BACKENDS = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']

def check_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def print_run_time(run_time):
    hour = run_time//3600
    minute = (run_time-3600*hour)//60
    second = run_time-3600*hour-60*minute
    print (f'The program run time ：{hour} hours {minute} minutes {second} seconds.')

class run_command_thread(threading.Thread):
    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID

    def run(self):
        try:
            while(True):
                command = file_video_queue.get(block=False, timeout=60)
                print(command)
                os.system(command)
                file_video_queue.task_done()
        except queue.Empty:
            pass
        except Exception as ex:
            print(f'The following exception occurs {type(ex)} : {ex}')
            print(traceback.format_exc())
        finally:
            print(self.threadID, ": done!")
            return 0

def transcode_video_to_image(video_path, img_path):
    for dir_path, dir_name_list, file_name_list in os.walk(video_path):
        for file_name in file_name_list:
            if(any(file_name.endswith(extension) for extension in ['.mp4'])):
                video_name = file_name[:-4]

                # create output path of images
                output_dir_path = os.path.join(img_path, video_name)
                check_make_dir(output_dir_path)

                command = (
                    f'ffmpeg -y -loglevel error -hide_banner -nostats '
                    f' -i {os.path.join(video_path, file_name)} '               # input file path
                    f' {os.path.join(output_dir_path, video_name)}_%04d.png'    # output file path, should be '#{name}_%04d.png'
                )
                print(command)
                os.system(command)

def transcode_video_to_image_multi_threads(video_path, img_path,  MAX_THREAD=8):
    for dir_path, dir_name_list, file_name_list in os.walk(video_path):
        for file_name in file_name_list:
            if(any(file_name.endswith(extension) for extension in ['.mp4'])):

                video_name = file_name[:-4]

                # create output path of images
                output_dir_path = os.path.join(img_path, video_name)
                check_make_dir(output_dir_path)

                command = (
                    f'ffmpeg -y -loglevel error -hide_banner -nostats '
                    f' -i {os.path.join(video_path, file_name)} '               # input file path
                    f' {os.path.join(output_dir_path, video_name)}_%04d.png'    # output file path, should be '#{name}_%04d.png'
                )
                file_video_queue.put(command)

    for i in range(MAX_THREAD):
        tmp_thread = run_command_thread(threadID=i)
        tmp_thread.start()

    file_video_queue.join() # Wait for all data to be processed
    print("all video is encoded!")

def get_gaussian_QP(QP_min=22, QP_max=38, QP_step=2, gaussian_mean=30, gaussian_variance=3):
    random_num = random.gauss(gaussian_mean, gaussian_variance)
    random_qp = round(random_num / QP_step) * QP_step
    while(random_qp < QP_min or random_qp > QP_max):
        random_num = random.gauss(gaussian_mean, gaussian_variance)
        random_qp = round(random_num / QP_step) * QP_step
    return random_qp

def get_uniform_QP(QP_min=22, QP_max=38, QP_step=2):
    qp_list = list(range(QP_min, QP_max+QP_step, QP_step))
    return random.choice(qp_list)

def downsample_video_multi_threads(video_path, output_video_path, MAX_THREAD=8):
    begin_time = time.time()
    for dir_path, dir_name_list, file_name_list in os.walk(video_path):
        for file_name in file_name_list:
            if(any(file_name.endswith(extension) for extension in ['.mp4'])):

                video_name = file_name[:-4]

                for down_sample_scale in [1, 2, 4]:
                    # create output path of videos
                    output_dir_path = os.path.join(output_video_path, 'down_x' + str(down_sample_scale))
                    check_make_dir(output_dir_path)

                    qp = get_gaussian_QP()

                    command = (
                        f'ffmpeg -y -loglevel error -hide_banner -nostats '
                        f' -i {os.path.join(video_path, file_name)}'                # input file path
                        f' -vf scale={1920//down_sample_scale}:{1080//down_sample_scale}' # resolution of output video, assumed all video is 1080P
                        f' -c:v libx264 -qp {str(qp)} '                             # qp
                        f' {os.path.join(output_dir_path, video_name)}_qp{str(qp)}_x{str(down_sample_scale)}.mp4'  # output file path, should be '#{name}_%04d.png'
                    )
                    file_video_queue.put(command)

    for i in range(MAX_THREAD):
        tmp_thread = run_command_thread(threadID=i)
        tmp_thread.start()

    file_video_queue.join() # Wait for all data to be processed
    print("all video is encoded!")
    end_time = time.time()
    print_run_time(round(end_time-begin_time))

def decode_video_to_tmp_dir(video_path, video_name, decode_frames_num=-1):
    output_raw_img_dir = os.path.join('/tmp/tmp_video', video_name)
    check_make_dir(output_raw_img_dir)
    if(decode_frames_num <= 0):
        decode_command = (
            f'ffmpeg -y -loglevel error -hide_banner -nostats '
            f' -i {video_path} '                                            # input file path
            f' {os.path.join(output_raw_img_dir, video_name)}_%04d.png'     # output file path, should be '#{name}_%04d.png'
        )
    else:
        decode_command = (
            f'ffmpeg -y -loglevel error -hide_banner -nostats '
            f' -i {video_path} '                                            # input file path
            f' -vframes {decode_frames_num} '                               # number of decoded frames
            f' {os.path.join(output_raw_img_dir, video_name)}_%04d.png'     # output file path, should be '#{name}_%04d.png'
        )
    os.system(decode_command)
    return output_raw_img_dir

def rm_video_dir(path):
    command = (f'rm -rf {path}')
    os.system(command)

"""
This function is based on https://github.com/ageitgey/face_recognition and https://github.com/serengil/deepface
To accelerate the infer process, you could install dlib with cuda (followed by https://gist.github.com/nguyenhoan1988/ed92d58054b985a1b45a521fcf8fa781)
"""
class get_face_box_thread(threading.Thread):
    def __init__(self, threadID, faces_locations_path, model="dlib", is_save_video=False):
        threading.Thread.__init__(self)
        assert(model in BACKENDS)
        self.threadID = threadID
        self.faces_locations_path = faces_locations_path
        self.model = model
        self.is_save_video = is_save_video
        if(self.model != 'dlib'):
            self.detector = FaceDetector.build_model(self.model)

    def get_face_location(self, faces_locations, index):
        assert(self.model in BACKENDS)
        if self.model == 'dlib':
            return faces_locations[index]
        else:
            face, region = faces_locations[index]
            left, top, width, height = region
            right = left + width
            bottom = height + top
            if(self.model in ['mtcnn', 'retinaface']):
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2
            return top, right, bottom, left

    def face_detect(self, img_path):
        assert(self.model in BACKENDS)
        if self.model == 'dlib':
            raw_image = face_recognition.load_image_file(img_path)
            faces_locations = face_recognition.face_locations(raw_image, number_of_times_to_upsample=0, model="cnn")
        else:
            raw_image = cv2.imread(img_path) # read as BGR
            if(self.model in ['mtcnn', 'retinaface']):
                down_scale_img = cv2.resize(raw_image, (raw_image.shape[1]//2, raw_image.shape[0]//2), interpolation=cv2.INTER_CUBIC) # reverse order
                faces_locations = FaceDetector.detect_faces(self.detector, self.model, down_scale_img, align=False)
            else:
                faces_locations = FaceDetector.detect_faces(self.detector, self.model, raw_image, align=False)
            raw_image = raw_image[:, :, ::-1] #bgr to rgb
        return raw_image, faces_locations

    def run(self):
        try:
            while(True):
                video_path = file_video_queue.get(block=False, timeout=60)
                video_name = os.path.split(video_path)[1][:-4]
                print(video_name, " begin!!")
                output_raw_img_dir = decode_video_to_tmp_dir(video_path, video_name)
                output_txt_path = os.path.join(self.faces_locations_path, video_name + ".json")

                output_json = []
                with open(output_txt_path, "w") as fp_out:
                    for _, _, file_name_list in os.walk(output_raw_img_dir): # every image
                        file_name_list.sort()
                        for file_name in file_name_list:
                            if(any(file_name.endswith(extension) for extension in ['.png'])):
                                tmp_output_json = {}
                                tmp_output_json['frame_index'] = file_name.split(".")[0][-4:]
                                if( int(tmp_output_json['frame_index']) > 100): # only detect previous 100 frames
                                    continue
                                img_path = os.path.join(output_raw_img_dir, file_name)
                                raw_image, faces_locations = self.face_detect(img_path)

                                faces_json = []
                                # process location
                                for index in range(len(faces_locations)):  # index-th face
                                    top, right, bottom, left = self.get_face_location(faces_locations, index)
                                    faces_json.append({'top': int(top), 'bottom':int(bottom), 'left':int(left), 'right':int(right)})

                                    # uncomments this code to visualize the face location
                                    # output_dir_path = os.path.join(self.faces_locations_path, video_name)
                                    # check_make_dir(output_dir_path)
                                    # save_img_file_path = os.path.join(output_dir_path, video_name + "_" + file_name[-8:])
                                    # pil_image = Image.fromarray(raw_image)
                                    # draw = ImageDraw.Draw(pil_image)
                                    # draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255), width=10) # draw the rectangle in the image
                                    # pil_image.save(save_img_file_path)
                                tmp_output_json['faces'] = faces_json
                                output_json.append(tmp_output_json)
                    json.dump(output_json, fp_out, indent=4)

                if(not self.is_save_video):
                    rm_video_dir(output_raw_img_dir)
                file_video_queue.task_done()
        except queue.Empty:
            pass
        except Exception as ex:
            print(f'The following exception occurs{type(ex)} : {ex}')
            print(traceback.format_exc())
        finally:
            print(self.threadID, ": done!")
            return 0

def get_face_box_multi_threads(video_dir_path, faces_locations_path, model='dlib', MAX_THREAD=8, overwrite=False, is_previous_file=True): # flag is_previous_file to divide all file to 2 GPu
    begin_time = time.time()
    assert(model in BACKENDS)
    for dir_path, dir_name_list, file_name_list in os.walk(video_dir_path):
        file_name_list.sort()
        total = len(file_name_list)
        for index in range(total): # every video
            video_name = file_name_list[index]
            if(is_previous_file):
                if(index > (total // 2)):
                    continue
            else:
                if(index <= (total // 2)):
                    continue
            video_path = os.path.join(video_dir_path, video_name)
            file_video_queue.put(video_path)

    print(f"begin to process with {MAX_THREAD} threads, len(file_video_queue) = {file_video_queue.qsize()}")
    thread_list = []
    for i in range(MAX_THREAD):
        tmp_thread = get_face_box_thread(threadID=i, faces_locations_path=faces_locations_path, model=model)
        tmp_thread.start()
        thread_list.append(tmp_thread)

    file_video_queue.join() # Wait for all data to be processed
    print("all video is processed!")
    end_time = time.time()
    print_run_time(round(end_time-begin_time))

def determine_crop_region(counter_np):
    ret_regions = []
    height, width = counter_np.shape
    MAX = np.max(counter_np)
    threshold = int(MAX*0.90)
    if(threshold > 20): # if threshold < 20, it is probably no face exists here
        for height_index in range(height):
            for width_index in range(width):
                if(counter_np[height_index, width_index] < threshold):
                    continue # do nothing
                if(in_the_region(height_index, width_index, ret_regions)):
                    continue # do nothing
                else:
                    ret_regions.append(get_face_region_by_dilation(counter_np, width_index=width_index, height_index=height_index))
    return ret_regions

def in_the_region(height_axis, width_axis, regions_list):
    for top, right, bottom, left in regions_list:
        if((top <= height_axis <= bottom) and (left <= width_axis <= right)):
            return True
    return False

def get_face_region_by_dilation(counter_np, width_index, height_index):
    top = bottom = height_index
    right = left = width_index
    height, width = counter_np.shape
    while((right-left) < width or (bottom-top) < height):
        left_value = right_value = top_value = bottom_value = 0 # init
        if(left > 0 and (right-left) < width):
            left_value = np.sum(counter_np[top:bottom+1,left-1])
            if(left_value > 0):
                left -= 1
        if(right < width - 1 and (right-left) < width):
            right_value = np.sum(counter_np[top:bottom+1,right:right+1])
            if(right_value > 0):
                right += 1

        if(top > 0 and (bottom-top) < height):
            top_value = np.sum(counter_np[top-1,left:right+1])
            if(top_value > 0):
                top -= 1
        if(bottom < height - 1 and (bottom-top) < height):
            bottom_value = np.sum(counter_np[bottom+1,left:right+1])
            if(bottom_value > 0):
                bottom += 1
        if(left_value == 0 and right_value == 0 and top_value == 0 and bottom_value == 0):
            return (top, right, bottom, left)
    return (top, right, bottom, left)

def enlarge_region_box(top, right, bottom, left, height=1080, width=1920, enlarge_ratio=1.5):
    assert(enlarge_ratio >= 1)
    new_width = int((right - left) * enlarge_ratio)
    new_height = int((bottom - top) * enlarge_ratio)
    delta_top_bottom = int((bottom - top) * (enlarge_ratio - 1) / 2)
    delta_left_right = int((right - left) * (enlarge_ratio - 1) / 2)

    if(new_width > (width-1)):
        ret_right = width-1
        ret_left = 0
    elif((left - delta_left_right) < 0):
        ret_left = 0
        ret_right = new_width
    elif((right + delta_left_right) > (width - 1)):
        ret_right = width-1
        ret_left = width - 1 - new_width
    else:
        ret_left = left - delta_left_right
        ret_right = right + delta_left_right

    if(new_height > (height-1)):
        ret_bottom = height-1
        ret_top = 0
    elif((top - delta_top_bottom) < 0):
        ret_top = 0
        ret_bottom = new_height
    elif((bottom + delta_top_bottom) > (height - 1)):
        ret_bottom = height-1
        ret_top = height - 1 - new_height
    else:
        ret_top = top - delta_top_bottom
        ret_bottom = bottom + delta_top_bottom

    return (ret_top, ret_right, ret_bottom, ret_left)

def enlarge_region_box_ignore_outside(top, right, bottom, left, height=1080, width=1920, enlarge_ratio=1.5):
    assert(enlarge_ratio >= 1)
    new_width = int((right - left) * enlarge_ratio)
    new_height = int((bottom - top) * enlarge_ratio)
    delta_top_bottom = int((bottom - top) * (enlarge_ratio - 1) / 2)
    delta_left_right = int((right - left) * (enlarge_ratio - 1) / 2)

    ret_left = left - delta_left_right
    ret_right = right + delta_left_right
    if(ret_left < 0):
        ret_left = 0
    if(ret_right > (width-1)):
        ret_right = width-1

    ret_top = top - delta_top_bottom
    ret_bottom = bottom + delta_top_bottom
    if(ret_top < 0):
        ret_top = 0
    if(ret_bottom > (height-1)):
        ret_bottom = height-1

    return (ret_top, ret_right, ret_bottom, ret_left)

def align_coordinates_to_multiples_4(regions_list):
    ret_regions_list = []
    for top, right, bottom, left in regions_list:
        ret_regions_list.append((
            (top // 4) * 4,
            math.ceil(right // 4) * 4,
            math.ceil(bottom // 4) * 4,
            (left // 4) *4
        ))
    return ret_regions_list

def remove_small_face(regions_list, threshold_size):
    ret_regions_list = []
    threshold_width, threshold_height = threshold_size
    for top, right, bottom, left in regions_list:
        if((right-left) < threshold_width and (bottom-top) < threshold_height):
            continue # discard small face
        else:
            ret_regions_list.append((top, right, bottom, left))
    return ret_regions_list

def enlarge_to_great_than_256(regions_list, height=1080, width=1920):
    ret_regions_list = []
    for top, right, bottom, left in regions_list:
        if (right - left) < 256:
            mid = (right + left) // 2
            left = mid - (256 // 2)
            right = mid + (256 // 2)
            if(left < 0):
                left = 0
                right = 256
            if(right > (width-1)):
                left = width-1-256
                right = width-1

        if (bottom - top) < 256:
            mid = (bottom + top) // 2
            top = mid - (256 // 2)
            bottom = mid + (256 // 2)
            if(top < 0):
                top = 0
                bottom = 256
            if(bottom > (height-1)):
                top = height-1-256
                bottom = height-1

        ret_regions_list.append((top, right, bottom, left))

    return ret_regions_list

def determine_crop_region_process(id, video_path, faces_locations_path, crop_face_path,
                            face_info_queue, is_save_crop_video=False):
    frame_start, frame_end = (1, 101)
    face_dict = {}
    try:
        video_name = os.path.split(video_path)[1][:-4]
        face_dict[video_name] = []

        if(is_save_crop_video):
            output_raw_img_dir = decode_video_to_tmp_dir(video_path, video_name)
            height, width, channels = cv2.imread(os.path.join(output_raw_img_dir, video_name + "_0001.png")).shape
        else:
            height = 1080 # assume all video is 1920x1080
            width = 1920
        counter_np = np.zeros((height, width), dtype="int32") # (1080, 1920)

        face_locations_json_path = os.path.join(faces_locations_path, video_name + ".json")
        flag_is_stop = False
        with open(face_locations_json_path, "r") as fp_in:
            load_list = json.load(fp_in) # load json (as list)
            aver_face_size_list = []
            for iter_dict in load_list:
                frame_index = iter_dict['frame_index']
                faces_locations = iter_dict['faces']

                for location_index in range(len(faces_locations)):
                    location = faces_locations[location_index]
                    if(len(aver_face_size_list) < len(faces_locations)):
                        aver_face_size_list.append([])
                    top, right, bottom, left = location['top'], location['right'], location['bottom'], location['left']
                    (top, right, bottom, left) = enlarge_region_box_ignore_outside(top=top, right=right, bottom=bottom, left=left, height=1080, width=1920, enlarge_ratio=2)
                    counter_np[top:bottom, left:right] += 1
                    aver_face_size_list[location_index].append((bottom - top) * (right - left))

                if( int(frame_index) == frame_end-1 ):
                    # determine the best crop region
                    regions = determine_crop_region(counter_np) # base region

                    # is large motion video?
                    for i in range(len(regions)):
                        top, right, bottom, left = regions[i]
                        region_size = (bottom - top) * (right - left)
                        if(np.mean(aver_face_size_list[i]) / region_size > 0.6 or int(frame_index) == 300):
                            flag_is_stop = True

                    # pre-process regions for VSR task
                    regions = remove_small_face(regions, threshold_size=(200,200))
                    regions = enlarge_to_great_than_256(regions, height=1080, width=1920)
                    regions = align_coordinates_to_multiples_4(regions)

                    for region in regions:
                        tmp_region_dic = {"top":region[0], "right":region[1], "bottom":region[2], "left":region[3]}
                        tmp_dict = {"face":tmp_region_dic, "frame_start":frame_start, "frame_end":frame_end}
                        face_dict[video_name].append(tmp_dict)

                    if(is_save_crop_video): # show crop region as save as imgs
                        for region_index in range(len(regions)):
                            top, right, bottom, left = regions[region_index]
                            output_dir_path = os.path.join(crop_face_path, video_name + '-' + str(region_index))
                            check_make_dir(output_dir_path)
                            for _, _, file_name_list in os.walk(output_raw_img_dir): # every image
                                file_name_list.sort()
                                for file_name in file_name_list:
                                    if(any(file_name.endswith(extension) for extension in ['.png'])):
                                        frame_index = int(file_name.split(".")[0][-4:])
                                        if(frame_start <= frame_index < frame_end):
                                            img_path = os.path.join(output_raw_img_dir, file_name)
                                            save_crop_img_path = os.path.join(output_dir_path, video_name + '-' + str(region_index) + "_" + file_name[-8:])
                                            raw_image = cv2.imread(img_path)[:, :, ::-1] # read as BGR, and convert to RGB
                                            pil_image = Image.fromarray(raw_image)
                                            draw = ImageDraw.Draw(pil_image)
                                            draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0), width=10)     # red
                                            pil_image.save(save_crop_img_path)

                            # encode as video
                            compress_command = (
                                f'ffmpeg -y -loglevel error -hide_banner -nostats -r 30 '
                                f' -i {os.path.join(output_dir_path, video_name)}-{str(region_index)}_%04d.png '
                                f' -vcodec libx264 -pix_fmt yuv420p '
                                f' {os.path.join(crop_face_path, video_name)}-{str(region_index)}.mp4'
                            )
                            os.system(compress_command)

                            rm_command = (f'rm -rf {output_dir_path}')
                            os.system(rm_command)

                    # update for next
                    frame_start = frame_end
                    frame_end = frame_end + 100

                if(flag_is_stop):
                    break
        if(len(face_dict[video_name]) > 0):
            face_info_queue.put(face_dict)
        if(is_save_crop_video):
            rm_video_dir(output_raw_img_dir)

    except Exception as ex:
        print("The following exception occurs", type(ex), ": ", ex)
        print(traceback.format_exc())
    finally:
        if(id % 100 ==0):
            print(f'processes {id}({video_name}) is done!')
        return 0

# MSRA server container 2 Xeon Gold 5118 Processor, each CPU have 12 core and can run 24 thread.
# To take full advantage of the server's performance, the args max_process is recommended to be set to 48
def determine_crop_region_multi_process(videos_dir_path, faces_locations_path, crop_face_path,
                                        max_process=48, is_save_crop_video=False):
    begin_time = time.time()
    num_video = 0
    manage = multiprocessing.Manager()
    face_info_queue = manage.Queue() # should use manage.Queue() in multipeocessing
    pool = multiprocessing.Pool(processes=max_process)
    for dir_path, dir_name_list, file_name_list in os.walk(videos_dir_path):
        file_name_list.sort()
        num_video = len(file_name_list)
        for i in range(num_video):  # every video
            video_name = file_name_list[i]
            video_path = os.path.join(videos_dir_path, video_name)
            pool.apply_async(determine_crop_region_process,
                    args=(i, video_path, faces_locations_path, crop_face_path, face_info_queue, is_save_crop_video, ))
    pool.close()
    pool.join()

    print("all the processes is done, begin to generate json.")
    with open(os.path.join(faces_locations_path, "_all_video_enlargex2_ingore-outside_" + str(num_video) + '.json'), "w") as fp_out:
        output_json = {}
        while not face_info_queue.empty():
            face_dict = face_info_queue.get(block=False)
            output_json.update(face_dict)
        assert(len(output_json) > 0)
        json.dump(output_json, fp_out, indent=4)

    end_time = time.time()
    print_run_time(round(end_time-begin_time))

def get_static_size(faces_locations_path):
    begin_time = time.time()

    min_face_size = min_width= min_height = 9999
    max_face_size = max_width = max_height = 0
    which_min = ""
    which_max = ""

    static_dict = {}
    WIDTH_MAX = 700
    HEIGHT_MAX = 900
    for width_iter in range(0, WIDTH_MAX, 100):
        for height_iter in range(0, HEIGHT_MAX, 100):
            static_dict[f'{str(height_iter)}x{str(width_iter)}'] = 0

    for _, _, file_name_list in os.walk(faces_locations_path):
        file_name_list.sort()
        for face_locations_file_name in file_name_list: # every video
            if("_" in face_locations_file_name):
                continue
            face_locations_json_path = os.path.join(faces_locations_path, face_locations_file_name)
            with open(face_locations_json_path, "r") as fp_in:
                load_list = json.load(fp_in) # load json (as list)
                for iter_dict in load_list:
                    frame_index = iter_dict['frame_index']
                    faces_locations = iter_dict['faces']
                    if(int(frame_index) > 100):
                        continue
                    for faces_locations_index in range(len(faces_locations)):
                        location = faces_locations[faces_locations_index]
                        top, right, bottom, left = location['top'], location['right'], location['bottom'], location['left']
                        width = right - left
                        height = bottom - top
                        face_size = width * height

                        # get the biggest face and minimun face
                        if(face_size > max_face_size):
                            max_face_size = face_size
                            max_width = width
                            max_height = height
                            which_max = face_locations_file_name + frame_index
                        if(face_size < min_face_size):
                            min_face_size = face_size
                            min_width = width
                            min_height = height
                            which_min = face_locations_file_name + frame_index

                        tmp_key = str(int((height // 100) * 100)) + 'x' + str(int((width // 100) * 100))
                        static_dict[tmp_key] += 1

    remove_key_list = []
    for key in static_dict.keys(): # can't change the size of dictionary during iteration of dictionary
        if(static_dict[key] == 0):
            remove_key_list.append(key)
    for key in remove_key_list:
        static_dict.pop(key)
    print(static_dict)
    csv_file_path = os.path.join(faces_locations_path, "statistical_results.csv")
    with open(csv_file_path,"w") as csv_file:
        writer = csv.DictWriter(csv_file, static_dict.keys())
        writer.writeheader()
        writer.writerow(static_dict)

    print(f'{which_max}: max_face_size={max_face_size}, max_width={max_width}, max_height={max_height}.')
    print(f'{which_min}: min_face_size={min_face_size}, min_width={min_width}, min_height={min_height}.')

    end_time = time.time()
    print_run_time(round(end_time-begin_time))

def get_video_fullname_by_prefix(video_dir_path, prefix):
    for dir_path, dir_name_list, file_name_list in os.walk(video_dir_path):
        for file_name in file_name_list:
            if(file_name.startswith(prefix)):
                return file_name[:-4]
    else:
        print(f'ERROR: do not found this video: {prefix}.')
        return None

def generate_vsr_dataset_process(processes_id, video_name, gt_videos_dir_path, down_video_dir_path, output_dir, faces_info_dict):
    try:
        # get the video path, including gt videos, downsample x2 and x4 videos.
        gt_video_path = os.path.join(gt_videos_dir_path, video_name + ".mp4")
        x1_videos_dir_path = os.path.join(down_video_dir_path, "down_x1")
        x1_video_name = get_video_fullname_by_prefix(x1_videos_dir_path, video_name)
        x1_video_path = os.path.join(x1_videos_dir_path, x1_video_name + ".mp4")
        x2_videos_dir_path = os.path.join(down_video_dir_path, "down_x2")
        x2_video_name = get_video_fullname_by_prefix(x2_videos_dir_path, video_name)
        x2_video_path = os.path.join(x2_videos_dir_path, x2_video_name + ".mp4")
        x4_videos_dir_path = os.path.join(down_video_dir_path, "down_x4")
        x4_video_name = get_video_fullname_by_prefix(x4_videos_dir_path, video_name)
        x4_video_path = os.path.join(x4_videos_dir_path, x4_video_name + ".mp4")

        # generate tmp dir to save raw images, which is used to crop later
        raw_gt_img_dir = decode_video_to_tmp_dir(gt_video_path, video_name)
        raw_x1_img_dir = decode_video_to_tmp_dir(x1_video_path, x1_video_name)
        raw_x2_img_dir = decode_video_to_tmp_dir(x2_video_path, x2_video_name)
        raw_x4_img_dir = decode_video_to_tmp_dir(x4_video_path, x4_video_name)

        face_count = 0
        for face_dict in faces_info_dict:
            frame_start = face_dict['frame_start']
            frame_end = face_dict['frame_end']
            location = face_dict['face']
            # generage output dir
            output_gt_dir_path = os.path.join(output_dir, 'gt', f'{video_name}-{face_count}')
            output_x1_dir_path = os.path.join(output_dir, 'down_x1', f'{x1_video_name}-{face_count}')
            output_x2_dir_path = os.path.join(output_dir, 'down_x2', f'{x2_video_name}-{face_count}')
            output_x4_dir_path = os.path.join(output_dir, 'down_x4', f'{x4_video_name}-{face_count}')
            check_make_dir(output_gt_dir_path)
            check_make_dir(output_x1_dir_path)
            check_make_dir(output_x2_dir_path)
            check_make_dir(output_x4_dir_path)
            top, right, bottom, left = location['top'], location['right'], location['bottom'], location['left']
            for frame_index in range(frame_start, frame_end):
                # crop gt images
                raw_gt_img_path = os.path.join(raw_gt_img_dir, f'{video_name}_{frame_index:04d}.png')
                gt_pil_image = Image.open(raw_gt_img_path)
                gt_pil_image = gt_pil_image.crop((left, top, right, bottom))
                crop_gt_img_path = os.path.join(output_gt_dir_path, f'{video_name}-{face_count}_{frame_index:04d}.png')
                gt_pil_image.save(crop_gt_img_path)

                # crop x1 images
                raw_x1_img_path = os.path.join(raw_x1_img_dir, f'{x1_video_name}_{frame_index:04d}.png')
                x1_pil_image = Image.open(raw_x1_img_path)
                x1_pil_image = x1_pil_image.crop((left, top, right, bottom))
                crop_x1_img_path = os.path.join(output_x1_dir_path, f'{x1_video_name}-{face_count}_{frame_index:04d}.png')
                x1_pil_image.save(crop_x1_img_path)

                # crop x2 images
                raw_x2_img_path = os.path.join(raw_x2_img_dir, f'{x2_video_name}_{frame_index:04d}.png')
                x2_pil_image = Image.open(raw_x2_img_path)
                x2_pil_image = x2_pil_image.crop((left//2, top//2, right//2, bottom//2))
                crop_x2_img_path = os.path.join(output_x2_dir_path, f'{x2_video_name}-{face_count}_{frame_index:04d}.png')
                x2_pil_image.save(crop_x2_img_path)

                # crop x4 images
                raw_x4_img_path = os.path.join(raw_x4_img_dir, f'{x4_video_name}_{frame_index:04d}.png')
                x4_pil_image = Image.open(raw_x4_img_path)
                x4_pil_image = x4_pil_image.crop((left//4, top//4, right//4, bottom//4))
                crop_x4_img_path = os.path.join(output_x4_dir_path, f'{x4_video_name}-{face_count}_{frame_index:04d}.png')
                x4_pil_image.save(crop_x4_img_path)

            face_count += 1

        rm_video_dir(raw_gt_img_dir)
        rm_video_dir(raw_x1_img_dir)
        rm_video_dir(raw_x2_img_dir)
        rm_video_dir(raw_x4_img_dir)
    except Exception as ex:
        print("The following exception occurs", type(ex), ": ", ex)
        print(traceback.format_exc())
    finally:
        if(processes_id % 100 ==0):
            print(f'processes {processes_id}({video_name}) is done!')
        return 0

def generate_vsr_dataset_multi_process(gt_videos_dir_path, down_video_dir_path, output_dir, info_json_path, max_process=48):
    begin_time = time.time()

    # generage output dir
    output_gt_dir_path = os.path.join(output_dir, 'gt')
    output_x1_dir_path = os.path.join(output_dir, 'down_x1')
    output_x2_dir_path = os.path.join(output_dir, 'down_x2')
    output_x4_dir_path = os.path.join(output_dir, 'down_x4')
    check_make_dir(output_gt_dir_path)
    check_make_dir(output_x1_dir_path)
    check_make_dir(output_x2_dir_path)
    check_make_dir(output_x4_dir_path)

    processes_num = 0
    pool = multiprocessing.Pool(processes=max_process)
    with open(info_json_path, "r") as fp_in:
        load_dict = json.load(fp_in) # load json (as list)
        for video_name in load_dict:
            pool.apply_async(generate_vsr_dataset_process,
                        args=(processes_num, video_name, gt_videos_dir_path, down_video_dir_path, output_dir, load_dict[video_name], ))
            processes_num += 1
    pool.close()
    pool.join()
    end_time = time.time()
    print_run_time(round(end_time-begin_time))

if __name__ == "__main__":
    fire.Fire()

# example
# python ./dataset_prepare.py transcode_video_to_image --video_path ../../data/huiguohe/deepfake_test/raw_video/ --img_path ../../data/huiguohe/deepfake_test/raw_pic/
# python ./dataset_prepare.py transcode_video_to_image_multi_threads --video_path ../../data/huiguohe/deepfake_test/raw_video/ --img_path ../../data/huiguohe/deepfake_test/raw_pic/ --MAX_THREAD 16

# python ./dataset_prepare.py downsample_video_multi_threads --video_path ../../data/huiguohe/deepfake_test/raw_video/ --output_video_path ../../data/huiguohe/deepfake_test/downsample_video/ --MAX_THREAD=8

# CUDA_VISIBLE_DEVICES=0 python ./dataset_prepare.py get_face_box_multi_threads --video_dir_path ../../data/huiguohe/deepfake_test/raw_video/ --faces_locations_path ../../data/huiguohe/deepfake_test/face_location/face_location_retinaface/ --MAX_THREAD 4 --model='retinaface' --is_previous_file=False
# CUDA_VISIBLE_DEVICES=1 python ./dataset_prepare.py get_face_box_multi_threads --video_dir_path ../../data/huiguohe/deepfake_test/raw_video/ --faces_locations_path ../../data/huiguohe/deepfake_test/face_location/face_location_retinaface/ --MAX_THREAD 4 --model='retinaface' --is_previous_file=True

# python ./dataset_prepare.py determine_crop_region_multi_process --videos_dir_path ../../data/huiguohe/deepfake_test/raw_video/ --faces_locations_path ../../data/huiguohe/deepfake_test/face_location/face_location_retinaface/ --crop_face_path ../../data/huiguohe/deepfake_test/crop_face/ --max_process 48

# python ./dataset_prepare.py get_static_size --faces_locations_path ../../data/huiguohe/deepfake_test/face_location/face_location_retinaface/

# python ./dataset_prepare.py generate_vsr_dataset_multi_process --gt_videos_dir_path ../../data/huiguohe/deepfake_test/raw_video/ --down_video_dir_path ../../data/huiguohe/deepfake_test/downsample_video/ --output_dir ../../data/huiguohe/deepfake_test/deepfake_vsr_dataset/ --info_json_path ../../data/huiguohe/deepfake_test/face_location/face_location_retinaface/_all_video_base_73.json --max_process 12
