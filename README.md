# Demo
Here's a GIF demonstrating the detection results when running at 35 fps on a Jetson Nano (fps has been cut to reduce GIF size).

![](https://github.com/ojalar/gifs/blob/main/modecla.gif)

# Dependencies
1. CUDA  

2. numpy  

3. OpenCV  

4. pandas  

5. PIL  

6. PyTorch  

7. skicit-learn  

8. TensorRT  

9. torchvision  

10. torch2trt

# Train
```
python3 train.py -tr <path to training .csv> -te <path to testing .csv> -w <name of saved weightfile>
```

## Data Format
Images should be listed in .csv files for training and testing, respectively. Each image should be given as a line:  
```
<path to image>, <class>
```
# Detection
```
python3 video_detection.py -p <path to video> -w <name of weightfile> -v <visual output (0 or 1)>
```
A sample clip is provided for demo purposes.

# Speed Benchmark
```
python3 speed_benchmark.py -p <path to video> -w <name of weightfile>
```

