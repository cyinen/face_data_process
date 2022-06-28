import os
import math
import random
import multiprocessing

def encode(idx, op):
    os.system(op)
    if idx % 100 == 0:
        print('Finish', idx)

if __name__ == '__main__':
    random.seed(19910511)

    # Train
    video_list = os.listdir('video_1080_train')
    video_list.sort()

    qp_list = list(range(24, 38, 2))

    pool = multiprocessing.Pool(processes=48)
    for i in range(len(video_list)):
        filename = os.path.splitext(video_list[i])[0]
        qp = random.choice(qp_list)
        for h in [90, 180, 270]:
            w = h // 90 * 160
            folder = 'video_lr%d_train' % h
            if not os.path.exists(folder):
                os.mkdir(folder)
            op = 'ffmpeg -loglevel error -hide_banner -nostats -i "video_1080_train/%s.mp4" -vf scale=%d:%d -c:v libx264 -qp %d "%s/%s_%02d.mp4"' % (filename, w, h, qp, folder, filename, qp)
            pool.apply_async(encode, args=(i, op))
    pool.close()
    pool.join()

    # Test
    video_list = os.listdir('video_1080_test')
    video_list.sort()

    qp_list = list(range(24, 38, 2))

    pool = multiprocessing.Pool(processes=48)
    for i in range(len(video_list)):
        filename = os.path.splitext(video_list[i])[0]
        for qp in qp_list:
            for h in [90, 180, 270]:
                w = h // 90 * 160
                folder = 'video_lr%d_test' % h
                if not os.path.exists(folder):
                    os.mkdir(folder)
                op = 'ffmpeg -loglevel error -hide_banner -nostats -i "video_1080_test/%s.mp4" -vf scale=%d:%d -c:v libx264 -qp %d "%s/%s_%02d.mp4"' % (filename, w, h, qp, folder, filename, qp)
                pool.apply_async(encode, args=(i, op))
    pool.close()
    pool.join()