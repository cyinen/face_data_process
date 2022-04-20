# -- coding: utf-8 --**

import threading
import os
import json
import fire
import queue
from PIL import Image, ImageDraw
from imageio import save
from matplotlib import widgets
import numpy as np
import cv2
import traceback
import re

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
                # with open(output_txt_path, "r") as fp_in:
                #     load_dict = json.load(fp_in)
                #     for iter in load_dict:
                #         print(iter)
                #         assert(1==2)
                # assert(1==2)
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

    def determine_crop_region(self, counter_np, square_size=512):
        ret_regions = []
        height, width = counter_np.shape
        MAX = np.max(counter_np)
        threshold = int(MAX*0.95)
        for height_index in range(height):
            for width_index in range(width):
                if(self.in_the_region(height_index, width_index, ret_regions)):
                    continue # do nothing
                if(counter_np[height_index, width_index] < threshold):
                    continue # do nothing
                else:
                    ret_regions.append(self.get_face_region_by_dilation(counter_np, height_index, width_index, square_size))
                    # print((height_index, width_index), ret_regions[-1])
        assert(1==2)
        return ret_regions

    def in_the_region(self, height_axis, width_axis, regions_list):
            for top, right, bottom, left in regions_list:
                if((top <= height_axis <= bottom) and (left <= width_axis <= right)):
                    return True
            return False

    def get_face_region_by_dilation(self, counter_np, width_index, height_index, square_size=512):
        top = bottom = height_index
        right = left = width_index
        height, width = counter_np.shape
        while((right-left) < square_size or (bottom-top) < square_size):
            if(left == 0):
                right = square_size
            if(right == height-1):
                left = width - 1 - square_size
            if(top == 0):
                bottom = square_size
            if(bottom == height-1):
                top = height - 1 - square_size

            if((right-left) < square_size):
                left_value = np.sum(counter_np[top:bottom,left-1])
                right_value = np.sum(counter_np[top:bottom,right+1])
                if(left_value < right_value):
                    right += 1
                elif(left_value > right_value):
                    left -= 1
                else:
                    if(right - left <= (square_size - 2)):
                        left -=1
                        right += 1
                    else:
                        left -= 1
            
            if((bottom-top) < square_size):
                top_value = np.sum(counter_np[top-1,left:right])
                bottom_value = np.sum(counter_np[bottom+1,left:right])
                if(top_value < bottom_value):
                    bottom += 1
                elif(top_value > bottom_value):
                    top -= 1
                else:
                    if(bottom - top <= (square_size - 2)):
                        top -=1
                        bottom += 1
                    else:
                        top -= 1
            # print((top, right, bottom, left))
        return (top, right, bottom, left)

    def run(self):
        try:
            while(True):
                video_path = file_video_queue.get(block=False, timeout=60)
                video_name = os.path.split(video_path)[1][:-4]
                print(video_name, " begin!!")

                output_raw_img_dir = decode_video_to_tmp_dir(video_path, video_name)
                face_locations_json_path = os.path.join(self.faces_locations_path, video_name + ".json")
                
                with open(face_locations_json_path, "r") as fp_in:
                    height, width, channels = cv2.imread(os.path.join(output_raw_img_dir, video_name + "_0001.png")).shape
                    counter_np = np.zeros((height, width), dtype="int32") # (1080, 1920)
                    load_list = json.load(fp_in) # load json (as list)
                    for iter_dict in load_list:
                        frame_index = iter_dict['frame_index']
                        faces_locations = iter_dict['faces']
                        if(int(frame_index) > 100):
                            continue
                        for location in faces_locations:   
                            top, right, bottom, left = location['top'], location['right'], location['bottom'], location['left']
                            counter_np[top:bottom, left:right] += 1

                # determine the best crop region
                regions = self.determine_crop_region(counter_np)

                # show crop region as save as imgs
                for region_index in range(len(regions)):
                    print(regions[region_index])
                    top, right, bottom, left = regions[region_index]
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
                                draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255), width=10)
                                pil_image.save(save_crop_img_path)
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
    for dir_path, dir_name_list, file_name_list in os.walk(videos_dir_path):
        file_name_list.sort()
        for video_name in file_name_list: # every video
            video_path = os.path.join(videos_dir_path, video_name)
            file_video_queue.put(video_path)

    print("begin to process with ", MAX_THREAD, " threads, len(file_video_queue) = ", file_video_queue.qsize())
    thread_list = []
    for i in range(MAX_THREAD):
        tmp_thread = determine_crop_region_thread(i, video_path, faces_locations_path, crop_face_path)
        tmp_thread.start()
        thread_list.append(tmp_thread)

    file_video_queue.join() # 等待所有的数据被处理完
    print("all video is crop")

if __name__ == "__main__":
    fire.Fire()
    # example
    # python ./dataset_prepare.py transcode_video_to_image --video_path ../../data/huiguohe/deepfake_test/raw_video/ --img_path ../../data/huiguohe/deepfake_test/raw_pic/
    # python ./dataset_prepare.py transcode_video_to_image_multi_threads --video_path ../../data/huiguohe/deepfake_test/raw_video/ --img_path ../../data/huiguohe/deepfake_test/raw_pic/ --MAX_THREAD 16
    # CUDA_VISIBLE_DEVICES=0 python ./dataset_prepare.py get_face_box_multi_threads --video_dir_path ../../data/huiguohe/deepfake_test/raw_video/ --faces_locations_path ../../data/huiguohe/deepfake_test/face_location/face_location_retinaface/ --MAX_THREAD 4 --model='retinaface' --is_previous_file=False
    # CUDA_VISIBLE_DEVICES=1 python ./dataset_prepare.py get_face_box_multi_threads --video_dir_path ../../data/huiguohe/deepfake_test/raw_video/ --faces_locations_path ../../data/huiguohe/deepfake_test/face_location/face_location_retinaface/ --MAX_THREAD 4 --model='retinaface' --is_previous_file=True
    # python ./dataset_prepare.py determine_crop_region_multi_threads --videos_dir_path ../../data/huiguohe/deepfake_test/raw_video/ --faces_locations_path ../../data/huiguohe/deepfake_test/face_location/face_location_retinaface/ --crop_face_path ../../data/huiguohe/deepfake_test/crop_face/ --MAX_THREAD 4