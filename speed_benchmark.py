import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18
import time
import sys
import argparse
from torch2trt import torch2trt

def benchmark(video_path, weight_file):
    # setup
    ##################
    net = resnet18(num_classes=3)
    device = torch.device('cuda')
    net.load_state_dict(torch.load(weight_file, map_location=lambda storage, loc: storage))
    net.to(device)
    net.eval()
    input_example = torch.ones([1, 3, 48, 48]).cuda()
    net_trt = torch2trt(net,[input_example], max_batch_size = 100)

    motion_img_w = 640
    motion_img_h = 360

    img_w = 1280
    img_h = 720
    #################

    # predict transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4786, 0.4712, 0.4665), (0.2352, 0.2317, 0.2367))
    ])

    # background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold = 8)
    fgbg.setBackgroundRatio(0.80)
    fgbg.setNMixtures(10)

    subtractor_warmup = 50
    warmup_counter = 0

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, orig = cap.read()
        if ret == False:
            break
        # initialize timer
        begin = time.time()
        # background subtraction
        frame = cv2.resize(orig,(motion_img_w, motion_img_h))
        fgmask = fgbg.apply(frame)

        if warmup_counter < subtractor_warmup:
            warmup_counter += 1
            continue

        # blur, threshold, erode, dilate
        fgmask_blur = cv2.GaussianBlur(fgmask,(5,5),0)
        thresh = cv2.threshold(fgmask_blur, 127, 256, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None,iterations = 1)
        thresh = cv2.dilate(thresh, None, iterations = 1)

        # contours to bounding box proposals
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]

        if len(cnts) == 0:
            continue

        regions = []
        boxes = []
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

        cnts = sorted(cnts, reverse = True, key = cv2.contourArea)[:100]

        for c in cnts:
            if cv2.contourArea(c) < 15:
                continue

            (x_motion, y_motion, w_motion, h_motion) = cv2.boundingRect(c)

            x = int(round(x_motion/motion_img_w * img_w))
            y = int(round(y_motion/motion_img_h * img_h))
            w = int(round(w_motion/motion_img_w * img_w))
            h = int(round(h_motion/motion_img_h * img_h))

            size = np.max((w,h))
            roi = np.zeros((size, size, 3), np.uint8)

            roi[int((size-h)/2):int((size-h)/2)+h,int((size-w)/2):int((size-w)/2)+w] = orig_rgb[y:y+h,x:x+w] # this needs optimization!!!!
            roi = cv2.resize(roi, (48,48))
            roi = transform(roi)

            boxes.append((x, y, w, h))
            regions.append(roi)

        if len(regions) == 0:
            continue

        # CNN inference
        with torch.no_grad():
            input = torch.stack(regions)
            input = input.to(device)
            output = net_trt(input)
            output = torch.nn.Softmax(1)(output)

        # clock the speed
        end = time.time() - begin
        print("fps:", 1/end)

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", help="path to a video")
ap.add_argument("-w", "--weights", help="weight file")
args = vars(ap.parse_args())

if not args.get("path", False):
        print("No path provided to video")
        sys.exit()
if not args.get("weights", False):
        print("No weight file provided")
        sys.exit()

benchmark(args.get("path"), args.get("weights"))
