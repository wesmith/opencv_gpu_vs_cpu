# opencv_gpu_vs_cpu

Compare speed of openCV algorithms on various platforms with CPU only and with GPU enabled.

Provide results in a spreadsheet.

First example is the Farnbacke optical-flow algorithm, comparing speeds of:

### jetson nano:         CPU
### jetson nano:         GPU-enabled opencv
### Acer Aspire-E5-576G: CPU
### Raspberry Pi 4:      CPU

# Summary from the spreadsheet

## Optical flow algorithm relative speeds:

### the Acer is 1.25 times as fast as the jetson nano GPU implementation

### the RPi4 is 0.49 times as fast as the jetson nano GPU implementation

### the jetson nano CPU is 0.37 times as fast as the jetson nano GPU implementation
#### ie, the RPi4 CPU is 0.49/0.37 = 1.32 times as fast as the jetson nano CPU implementation

## Full pipline relative speeds:

### the Acer is 1.43 times as fast as the jetson nano GPU implementation

### the RPi4 is 0.52 times as fast as the jetson nano GPU implementation

### the jetson nano CPU is 0.41 times as fast as the jetson nano GPU implementation
#### ie, the RPi4 CPU is 0.52/0.41 = 1.27 times as fast as the jetson nano CPU implementation

# Caveats

## This is not an in-depth analysis: out-of-the box SW and hardware that a casual hobbyist would utilize were used to perform these tests to get a feel for relative speeds. They do not represent a significant effort in optimizing anything.

### Perhaps a hand-tooled CUDA implementation of specific openCV algorithms would perform better than what was demonstrated here with the one tested algorithm, the Farneback optical flow.

## The C++ version of CUDA opencv was not investigated, only the python implementation was used.

# NOTEs

## Surprisingly the latest jetpack 4.6.3 seemed to be slower than jetpack 4.3, so jetpack 4.3 was used in these experiments (eg, jetpack 4.6.3 only appeared to use two threads while building opencv 4.7.0 with CUDA, taking 7 hours for the build, whereas jetpack 4.3 used all four threads and took only 5 hours for the build): this was not investigated further).

# To save build time, the openCV CUDA build on the jetson nano was limited to Python3 (Python2 turned off) and only CUDA_ARCH_BIN 5.3 (for the nano) was built to save time. CUDNN_VERSION='7.6' was used for consistency with jetpack 4.3.


 

