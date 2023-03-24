#!/usr/bin/python3

# WESmith 03/22/23

# clahe_test_GPU_jetson_nano.py
# test opencv 4.7.0 with cuda on jetson nano
# following the examples at
# https://learnopencv.com/getting-started-opencv-cuda-module/

# testing CLAHE: contrast limited adaptive histogram equalization

import cv2
import os

print('OpenCV version {}'.format(cv2.__version__))

img_path = '/usr/share/visionworks/sources/data'
img_name = 'lena.jpg'

img = cv2.imread(os.path.join(img_path, img_name), cv2.IMREAD_GRAYSCALE)
src = cv2.cuda_GpuMat()
src.upload(img)
 
clahe = cv2.cuda.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
dst   = clahe.apply(src, cv2.cuda_Stream.Null())
 
result = dst.download()
 
cv2.imshow("result", result)
cv2.waitKey(0)



