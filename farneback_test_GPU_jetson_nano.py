#!/usr/bin/python3

# WESmith 03/23/23

# farneback_test_GPU_jetson_nano.py
# test opencv 4.7.0 with cuda on jetson nano
# following the examples at
# https://learnopencv.com/getting-started-opencv-cuda-module/

# testing Farneback optical flow with GPU to get GPU times,
# compare to farneback_test_CPU.py

import cv2
import numpy as np
import os
import time

print('OpenCV version {}'.format(cv2.__version__))

scale  = 1.0  # factor to reduce the image

timers = {'reading':[],
          'pre-process':[],
          'optical flow':[],
          'post-process':[],
          'full pipeline':[]}

vid_path = '/usr/share/visionworks/sources/data'
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

if ret:

    frame = cv2.resize(prev_frame, (width, height))

    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)

    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gpu_prev = cv2.cuda_GpuMat()
    gpu_prev.upload(prev_frame)

    gpu_hsv    = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC3)
    gpu_hsv_8u = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_8UC3)

    gpu_h = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC1)
    gpu_s = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC1)
    gpu_v = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC1)    

    # set saturation to 1
    gpu_s.upload(np.ones_like(prev_frame, np.float32))


while True:

    start_full_time = time.time()

    start_read_time = time.time()

    ret, frame = cap.read()
    gpu_frame.upload(frame)

    end_read_time = time.time()

    timers['reading'].append(end_read_time - start_read_time)

    if not ret:
        break

    start_pre_time = time.time()

    gpu_frame = cv2.cuda.resize(gpu_frame, (width, height))

    gpu_current = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)

    end_pre_time = time.time()

    timers['pre-process'].append(end_pre_time - start_pre_time)

    start_of = time.time()

    gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(
               5, 0.5, False, 15, 3, 5, 1.2, 0)

    gpu_flow = cv2.cuda_FarnebackOpticalFlow.calc(
               gpu_flow, gpu_prev, gpu_current, None)
    
    end_of = time.time()

    timers['optical flow'].append(end_of - start_of)

    start_post_time = time.time()

    gpu_flow_x = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
    gpu_flow_y = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
    cv2.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y])

    gpu_mag, gpu_angle = cv2.cuda.cartToPolar(
                         gpu_flow_x, gpu_flow_y, angleInDegrees=True)

    gpu_v = cv2.cuda.normalize(gpu_mag, 0.0, 1.0, cv2.NORM_MINMAX, -1)

    angle = gpu_angle.download()
    angle *= (1 / 360.0) * (180 / 255.0)
    
    # set hue
    gpu_h.upload(angle)

    # merge h,s,v channels
    cv2.cuda.merge([gpu_h, gpu_s, gpu_v], gpu_hsv)

    # multiply by 255, convert to unsigned int
    gpu_hsv.convertTo(cv2.CV_8U, 255.0, gpu_hsv_8u, 0.0)

    gpu_bgr = cv2.cuda.cvtColor(gpu_hsv_8u, cv2.COLOR_HSV2BGR)

    frame = gpu_frame.download()

    bgr = gpu_bgr.download()

    gpu_prev = gpu_current
        
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

