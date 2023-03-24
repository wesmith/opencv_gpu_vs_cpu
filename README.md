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



