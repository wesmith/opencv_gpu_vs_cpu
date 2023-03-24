#!/home/smithw/.venv_openCV/bin/python3

# WESmith 03/23/23  modified for running on the acer

# farneback_test_CPU_acer.py
# test opencv 4.7.0 times on a particular problem
# following the examples at
# https://learnopencv.com/getting-started-opencv-cuda-module/

# testing Farneback optical flow with CPU to get CPU times,
# compare to farneback_test_GPU_jetson_nano.py

import cv2
import numpy as np
import os
import time

print('OpenCV version {}'.format(cv2.__version__))

scale  = 1.0  # factor to reduce the image
device = 'cpu'

timers = {'reading':[],
          'pre-process':[],
          'optical flow':[],
          'post-process':[],
          'full pipeline':[]}

#vid_path = '/usr/share/visionworks/sources/data'
vid_path = './'
vid_name = 'pedestrians.mp4'

cap = cv2.VideoCapture(os.path.join(vid_path, vid_name))

fps = cap.get(cv2.CAP_PROP_FPS)

num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

width  = int(width * scale)
height = int(height * scale)

txt = 'video {} is {} x {} at {} fps with {} frames'.\
      format(vid_name, width, height, fps, num_frames)
print(txt)

ret, prev_frame = cap.read()

if device == 'cpu':

    if ret:

        frame = cv2.resize(prev_frame, (width, height))

        prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        hsv = np.zeros_like(frame, np.float32)

        # set saturation to 1
        hsv[..., 1] = 1.0

while True:

    start_full_time = time.time()

    start_read_time = time.time()

    ret, frame = cap.read()

    end_read_time = time.time()

    timers['reading'].append(end_read_time - start_read_time)

    if not ret:
        break

    start_pre_time = time.time()

    frame = cv2.resize(frame, (width, height))

    current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    end_pre_time = time.time()

    timers['pre-process'].append(end_pre_time - start_pre_time)

    start_of = time.time()

    flow = cv2.calcOpticalFlowFarneback(prev_frame, current_frame, None,
                                        0.5, 5, 15, 3, 5, 1.2, 0)
    end_of = time.time()

    timers['optical flow'].append(end_of - start_of)

    start_post_time = time.time()

    mag, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1],
                                 angleInDegrees=True)
    # set hue
    hsv[..., 0] = angle * ((1 / 360.0) * (180 / 255.0))

    # set value
    hsv[..., 2] = cv2.normalize(mag, None, 0.0, 1.0, cv2.NORM_MINMAX, -1)

    hsv_8u = np.uint8(hsv * 255.0)

    bgr = cv2.cvtColor(hsv_8u, cv2.COLOR_HSV2BGR)

    prev_frame = current_frame

    end_post_time = time.time()

    timers['post-process'].append(end_post_time - start_post_time)
    
    end_full_time = time.time()

    timers['full pipeline'].append(end_full_time - start_full_time)

    cv2.imshow("original", frame)
    cv2.imshow("result", bgr)
    k = cv2.waitKey(1)
    if k == 27:
        break


print("Elapsed time")
for stage, seconds in timers.items():
    print("-", stage, ": {:0.3f} seconds".format(sum(seconds)))
 
# calculate frames per second
print("Default video FPS : {:0.3f}".format(fps))
 
of_fps = (num_frames - 1) / sum(timers["optical flow"])
print("Optical flow FPS : {:0.3f}".format(of_fps))
 
full_fps = (num_frames - 1) / sum(timers["full pipeline"])
print("Full pipeline FPS : {:0.3f}".format(full_fps))

