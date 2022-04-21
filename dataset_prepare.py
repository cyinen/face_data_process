# -- coding: utf-8 --**

import threading
import os
import json
import fire
import queue
from PIL import Image, ImageDraw
import numpy as np
import cv2
import traceback
import time
import csv

import face_recognition
from deepface import DeepFace
from deepface.detectors import FaceDetector
from pyparsing import matchOnlyAtCol

file_video_queue = queue.Queue()
face_info_queue = queue.Queue()
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']

def transcode_video_to_image(video_path, img_path):
    
    for dir_path, dir_name_list, file_name_list in os.walk(video_path):
        for file_name in file_name_list:
            if(any(file_name.endswith(extension) for extension in ['.mp4'])):

                video_name = file_name[:-4]

                # create output path of images
                output_dir_path = os.path.join(img_path, video_name)
                if not os.path.exists(output_dir_path):
                    os.makedirs(output_dir_path)

                command = ('ffmpeg -y ' + 
                          ' -i ' + os.path.join(video_path, file_name) + ' ' +               # input file path
                          os.path.join(output_dir_path, video_name) + '_%04d.png'            # output file path, should be '#{name}_%04d.png'
                )
                print(command)
                os.system(command)

class transcode_video_to_image_thread(threading.Thread):
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
        except Exception as ex:
            print("出现如下异常", type(ex), ": ", ex)
            print(traceback.format_exc())
        finally:
            print(self.threadID, ": done!")
            return 0

def transcode_video_to_image_multi_threads(video_path, img_path,  MAX_THREAD=8):
    
    for dir_path, dir_name_list, file_name_list in os.walk(video_path):
        for file_name in file_name_list:
            if(any(file_name.endswith(extension) for extension in ['.mp4'])):

                video_name = file_name[:-4]

                # create output path of images
                output_dir_path = os.path.join(img_path, video_name)
                if not os.path.exists(output_dir_path):
                    os.makedirs(output_dir_path)

                command = ('ffmpeg -y ' + 
                          ' -i ' + os.path.join(video_path, file_name) + ' ' +               # input file path
                          os.path.join(output_dir_path, video_name) + '_%04d.png'            # output file path, should be '#{name}_%04d.png'
                )
                file_video_queue.put(command)
                
    thread_list = []
    for i in range(MAX_THREAD):
        tmp_thread = transcode_video_to_image_thread(threadID=i)
        tmp_thread.start()
        thread_list.append(tmp_thread)

    file_video_queue.join() # 等待所有的数据被处理完
    print("all video is encoded!")

def decode_video_to_tmp_dir(video_path, video_name):
    output_raw_img_dir = os.path.join('/tmp/tmp_video', video_name)
    if not os.path.exists(output_raw_img_dir):
        os.makedirs(output_raw_img_dir)
    decode_command = ('ffmpeg -y ' + 
        ' -i ' + video_path + ' ' +                                            # input file path
        os.path.join(output_raw_img_dir, video_name) + '_%04d.png'                  # output file path, should be '#{name}_%04d.png'
    )
    os.system(decode_command)
    return output_raw_img_dir

def rm_video_dir(path):
    command = ('rm -rf ' + path)
    os.system(command)

"""
This function is based on https://github.com/ageitgey/face_recognition and https://github.com/serengil/deepface
To accelerate the infer process, you could install dlib with cuda (followed by https://gist.github.com/nguyenhoan1988/ed92d58054b985a1b45a521fcf8fa781)

"""
class get_face_box_thread(threading.Thread):
    def __init__(self, threadID, faces_locations_path, model="dlib", is_save_video=False):
        threading.Thread.__init__(self)
        assert(model in backends)
        self.threadID = threadID
        self.faces_locations_path = faces_locations_path
        self.model = model
        self.is_save_video = is_save_video
        if(self.model != 'dlib'):
            self.detector = FaceDetector.build_model(self.model)

    def get_face_location(self, faces_locations, index):
        assert(self.model in backends)
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
        assert(self.model in backends)
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
                with open(output_txt_path, "w") as fp_out:
                    output_json = []
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
                                    # if not os.path.exists(output_dir_path):
                                    #     os.makedirs(output_dir_path)
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
            print("出现如下异常", type(ex), ": ", ex)
            print(traceback.format_exc())
        finally:
            print(self.threadID, ": done!")
            return 0

def get_face_box_multi_threads(video_dir_path, faces_locations_path, model='dlib', MAX_THREAD=8, overwrite=False, is_previous_file=True): # flag is_previous_file to divide all file to 2 GPu
    assert(model in backends)
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

    print("begin to process with ", MAX_THREAD, " threads, len(file_video_queue) = ", file_video_queue.qsize())
    thread_list = []
    for i in range(MAX_THREAD):
        tmp_thread = get_face_box_thread(threadID=i, faces_locations_path=faces_locations_path, model=model)
        tmp_thread.start()
        thread_list.append(tmp_thread)

    file_video_queue.join() # 等待所有的数据被处理完
    print("all video is processed!")

class determine_crop_region_thread(threading.Thread):
    def __init__(self, threadID, video_path, faces_locations_path, crop_face_path):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.video_path = video_path
        self.faces_locations_path = faces_locations_path
        self.crop_face_path = crop_face_path

    def determine_crop_region(self, counter_np):
        ret_regions = []
        height, width = counter_np.shape
        MAX = np.max(counter_np)
        threshold = int(MAX*0.95)
        for height_index in range(height):
            for width_index in range(width):
                if(counter_np[height_index, width_index] < threshold):
                    continue # do nothing
                if(self.in_the_region(height_index, width_index, ret_regions)):
                    continue # do nothing
                else:
                    ret_regions.append(self.get_face_region_by_dilation(counter_np, width_index=width_index, height_index=height_index))
        return ret_regions

    def in_the_region(self, height_axis, width_axis, regions_list):
            for top, right, bottom, left in regions_list:
                if((top <= height_axis <= bottom) and (left <= width_axis <= right)):
                    return True
            return False

    def get_face_region_by_dilation(self, counter_np, width_index, height_index):
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

    def enlarge_region_box(self, top, right, bottom, left, height=1080, width=1920, enlarge_ratio=1.5):
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
    def run(self):
        try:
            while(True):
                video_path = file_video_queue.get(block=False, timeout=60)
                video_name = os.path.split(video_path)[1][:-4]
                print(video_name, " begin!!")

                output_raw_img_dir = decode_video_to_tmp_dir(video_path, video_name)
                face_locations_json_path = os.path.join(self.faces_locations_path, video_name + ".json")
                height, width, channels = cv2.imread(os.path.join(output_raw_img_dir, video_name + "_0001.png")).shape
                counter_np = np.zeros((height, width), dtype="int32") # (1080, 1920)

                with open(face_locations_json_path, "r") as fp_in:
                    load_list = json.load(fp_in) # load json (as list)
                    for iter_dict in load_list:
                        frame_index = iter_dict['frame_index']
                        faces_locations = iter_dict['faces']
                        if(int(frame_index) > 100):
                            continue
                        for location in faces_locations:   
                            top, right, bottom, left = location['top'], location['right'], location['bottom'], location['left']
                            (top, right, bottom, left) = self.enlarge_region_box(top=top, right=right, bottom=bottom, left=left, height=1080, width=1920, enlarge_ratio=2)
                            counter_np[top:bottom, left:right] += 1

                # determine the best crop region
                regions = self.determine_crop_region(counter_np)
                face_info_queue.put((video_name, regions))

                # show crop region as save as imgs
                for region_index in range(len(regions)):
                    top, right, bottom, left = regions[region_index]
                    # (top, right, bottom, left) = self.enlarge_region_box(top=top, right=right, bottom=bottom, left=left, height=1080, width=1920, enlarge_ratio=1.5)
                    output_dir_path = os.path.join(self.crop_face_path, video_name + '-' + str(region_index))
                    if not os.path.exists(output_dir_path):
                        os.makedirs(output_dir_path)
                    for _, _, file_name_list in os.walk(output_raw_img_dir): # every image
                        file_name_list.sort()
                        for file_name in file_name_list:
                            if(any(file_name.endswith(extension) for extension in ['.png'])):
                                frame_index = int(file_name.split(".")[0][-4:])
                                if(frame_index > 100):
                                    continue

                                img_path = os.path.join(output_raw_img_dir, file_name)
                                save_crop_img_path = os.path.join(output_dir_path, video_name + '-' + str(region_index) + "_" + file_name[-8:])
                                raw_image = cv2.imread(img_path)[:, :, ::-1] # read as BGR, and convert to RGB
                                pil_image = Image.fromarray(raw_image)
                                draw = ImageDraw.Draw(pil_image)
                                # draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255), width=10)   # blue
                                # draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=10)   # green
                                draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0), width=10)     # red
                                pil_image.save(save_crop_img_path)

                    # encode as video
                    compress_command = ('ffmpeg -y -r 30 ' +
                                ' -i ' + os.path.join(output_dir_path, video_name + '-' + str(region_index) + "_%04d.png") + " " +
                                '-vcodec libx264 -pix_fmt yuv420p ' +
                                os.path.join(self.crop_face_path, video_name + '-' + str(region_index) + '.mp4'))
                    os.system(compress_command)

                    rm_command = ('rm -rf ' + output_dir_path)
                    os.system(rm_command)

                rm_video_dir(output_raw_img_dir)

                file_video_queue.task_done()
        except queue.Empty:
            pass
        except Exception as ex:
            print("出现如下异常", type(ex), ": ", ex)
            print(traceback.format_exc())
        finally:
            print(self.threadID, ": done!")
            return 0

def determine_crop_region_multi_threads(videos_dir_path, faces_locations_path, crop_face_path, MAX_THREAD=8):
    begin_time = time.time()
    num_video = 0
    for dir_path, dir_name_list, file_name_list in os.walk(videos_dir_path):
        file_name_list.sort()
        num_video = len(file_name_list)
        for video_name in file_name_list: # every video
            video_path = os.path.join(videos_dir_path, video_name)
            file_video_queue.put(video_path)

    print("begin to process with ", MAX_THREAD, " threads, len(file_video_queue) = ", file_video_queue.qsize())
    for i in range(MAX_THREAD):
        tmp_thread = determine_crop_region_thread(i, video_path, faces_locations_path, crop_face_path)
        tmp_thread.start()

    file_video_queue.join() # 等待所有的数据被处理完
    with open(os.path.join(faces_locations_path, "all_video_base_" + str(num_video) + '.json'), "w") as fp_out:
        output_json = []
        while not face_info_queue.empty():
            video_name, regions = face_info_queue.get(block=False, timeout=60)
            tmp_output_json = {}
            faces_json = []
            for top, right, bottom, left in regions:
                faces_json.append({'top': int(top), 'bottom':int(bottom), 'left':int(left), 'right':int(right)})

            tmp_output_json[video_name] = faces_json
            output_json.append(tmp_output_json)
        json.dump(output_json, fp_out, indent=4)

    end_time = time.time()
    run_time = round(end_time-begin_time)
    hour = run_time//3600
    minute = (run_time-3600*hour)//60
    second = run_time-3600*hour-60*minute
    print (f'该程序运行时间：{hour}小时{minute}分钟{second}秒')

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
            static_dict[str(height_iter) + 'x' + str(width_iter)] = 0

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
    for key in static_dict.keys(): # can't change the size of dictionary during iteration of them
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
    run_time = round(end_time-begin_time)
    hour = run_time//3600
    minute = (run_time-3600*hour)//60
    second = run_time-3600*hour-60*minute
    print (f'该程序运行时间：{hour}小时{minute}分钟{second}秒.')

if __name__ == "__main__":
    fire.Fire()
    # example
    # python ./dataset_prepare.py transcode_video_to_image --video_path ../../data/huiguohe/deepfake_test/raw_video/ --img_path ../../data/huiguohe/deepfake_test/raw_pic/
    # python ./dataset_prepare.py transcode_video_to_image_multi_threads --video_path ../../data/huiguohe/deepfake_test/raw_video/ --img_path ../../data/huiguohe/deepfake_test/raw_pic/ --MAX_THREAD 16
    # CUDA_VISIBLE_DEVICES=0 python ./dataset_prepare.py get_face_box_multi_threads --video_dir_path ../../data/huiguohe/deepfake_test/raw_video/ --faces_locations_path ../../data/huiguohe/deepfake_test/face_location/face_location_retinaface/ --MAX_THREAD 4 --model='retinaface' --is_previous_file=False
    # CUDA_VISIBLE_DEVICES=1 python ./dataset_prepare.py get_face_box_multi_threads --video_dir_path ../../data/huiguohe/deepfake_test/raw_video/ --faces_locations_path ../../data/huiguohe/deepfake_test/face_location/face_location_retinaface/ --MAX_THREAD 4 --model='retinaface' --is_previous_file=True
    # python ./dataset_prepare.py determine_crop_region_multi_threads --videos_dir_path ../../data/huiguohe/deepfake_test/raw_video/ --faces_locations_path ../../data/huiguohe/deepfake_test/face_location/face_location_retinaface/ --crop_face_path ../../data/huiguohe/deepfake_test/crop_face/ --MAX_THREAD 4
    # python ./dataset_prepare.py get_static_size --faces_locations_path ../../data/huiguohe/deepfake_test/face_location/face_location_retinaface/