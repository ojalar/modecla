import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18
import argparse
import sys

def detection(video_path, weight_file, visual, show_misc, threshold):
    # setup
    ##################
    net = resnet18(num_classes=3)
    device = torch.device('cuda')

    net.load_state_dict(torch.load(weight_file, map_location=lambda storage, loc: storage))
    net.to(device)
    net.eval()

    motion_img_w = 640
    motion_img_h = 360

    img_w = 1280
    img_h = 720

    font = cv2.FONT_HERSHEY_SIMPLEX
    #################

    # predict transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4786, 0.4712, 0.4665), (0.2352, 0.2317, 0.2367))
    ])

    #background subtractor
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

        orig = cv2.resize(orig, (1280, 720))

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
        cnts = cnts[0]

        if len(cnts) == 0:
            continue

        regions = []
        boxes = []
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

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
            roi[int((size-h)/2):int((size-h)/2)+h,int((size-w)/2):int((size-w)/2)+w] = orig_rgb[y:y+h,x:x+w]
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
            output = net(input)
            output = torch.nn.Softmax(1)(output)
            values, indices = torch.max(output, 1)

        # if visual output enabled, show image result and print output
        # otherwise only print output
        print("FRAME -------------")
        if visual == True:
            for i in range(len(indices)):
                if indices[i] == 1 and values[i].item() >= threshold:
                    cv2.rectangle(orig, (boxes[i][0], boxes[i][1]), (boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]), (255,255,0), 2)
                    cv2.putText(orig,"Person",(boxes[i][0]-5, boxes[i][1]-10), font, 0.65,(0,0,0),3,cv2.LINE_AA)
                    cv2.putText(orig,"Person",(boxes[i][0]-5, boxes[i][1]-10), font, 0.65,(255,255,255),1,cv2.LINE_AA)
                    object = ["Person", str(values[i].item()), str((boxes[i][0] + boxes[i][2]/2)/img_w),
                                str((boxes[i][1] + boxes[i][3]/2)/img_h), str(boxes[i][2]/img_w), str(boxes[i][3]/img_h)]
                    print(" ".join(object))
                elif indices[i] == 2 and values[i].item() >= threshold:
                    cv2.rectangle(orig, (boxes[i][0], boxes[i][1]), (boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]), (0, 0, 255), 2)
                    cv2.putText(orig,"Car",(boxes[i][0]-5, boxes[i][1]-10), font, 0.65,(0,0,0),3,cv2.LINE_AA)
                    cv2.putText(orig,"Car",(boxes[i][0]-5, boxes[i][1]-10), font, 0.65,(255,255,255),1,cv2.LINE_AA)
                    object = ["Car", str(values[i].item()), str((boxes[i][0] + boxes[i][2]/2)/img_w),
                                str((boxes[i][1] + boxes[i][3]/2)/img_h), str(boxes[i][2]/img_w), str(boxes[i][3]/img_h)]
                    print(" ".join(object))
                elif show_misc:
                    cv2.rectangle(orig, (boxes[i][0], boxes[i][1]), (boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]), (255, 0, 0), 2)
                    cv2.putText(orig,"Misc",(boxes[i][0]-5, boxes[i][1]-10), font, 0.65,(0,0,0),3,cv2.LINE_AA)
                    cv2.putText(orig,"Misc",(boxes[i][0]-5, boxes[i][1]-10), font, 0.65,(255,255,255),1,cv2.LINE_AA)

            cv2.imshow("MoDeCla", orig)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            for i in range(len(indices)):
                if indices[i] == 1 and values[i].item() >= threshold:
                    object = ["Person", str(values[i].item()), str((boxes[i][0] + boxes[i][2]/2)/img_w),
                                str((boxes[i][1] + boxes[i][3]/2)/img_h), str(boxes[i][2]/img_w), str(boxes[i][3]/img_h)]
                    print(" ".join(object))
                elif indices[i] == 2 and values[i].item() >= threshold:
                    object = ["Car", str(values[i].item()), str((boxes[i][0] + boxes[i][2]/2)/img_w),
                                str((boxes[i][1] + boxes[i][3]/2)/img_h), str(boxes[i][2]/img_w), str(boxes[i][3]/img_h)]
                    print(" ".join(object))

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", help="path to a video")
ap.add_argument("-w", "--weights", help="weight file")
ap.add_argument("-v", "--visual", help="enables visual output, 1 or 0 (default is 1)")
ap.add_argument("-sm", "--show_misc", help="show misc detections in visual output, 1 or 0 (default is 0)")
ap.add_argument("-t", "--threshold", help="detection threshold, 0.33 to 1 (default is 0.5)")
args = vars(ap.parse_args())

if not args.get("path", False):
        print("No path provided to video")
        sys.exit()
if not args.get("weights", False):
        print("No weight file provided")
        sys.exit()
if not args.get("visual", False):
    visual = True
else:
    visual = bool(int(args.get("visual")))

if not args.get("show_misc", False):
    show_misc = False
else:
    show_misc = bool(int(args.get("show_misc")))

if not args.get("threshold", False):
    threshold = 0.5
else:
    threshold = float(args.get("threshold"))

detection(args.get("path"), args.get("weights"), visual, show_misc, threshold)
