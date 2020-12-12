from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from utils import *
import argparse
import os 
import os.path as osp
from darknet53 import Darknet
import pickle as pkl
import pandas as pd
import random

def arg_parse():
    parser = argparse.ArgumentParser(description='Hawkai Detection Module')
   
    parser.add_argument("--images", type = str, default = None, dest = 'images', help = "Input Images Directory" )
    parser.add_argument("--video", type = str, default = None, dest = 'videofile', help = "Input Video Directory" )
    parser.add_argument("--camera", type = int, default = 0, dest = 'camera', help = "Camera Input from" )
    parser.add_argument("--det", type = str, default = "test/images/image_detections", dest = 'detection', help = "Detection Results Directory" )
    parser.add_argument("--bs", type = int, default = 1, dest = "batch_size", help = "Batch size")
    parser.add_argument("--conf", type = float, default = 0.5, dest = "confidence", help = "Object Confidence to filter predictions")
    parser.add_argument("--nms_thresh", type = float, default = 0.2, dest = "nms_thresh", help = "NMS Threshhold")
    parser.add_argument("--cfg", type = str, default = "config/yolo_coco.cfg", dest = 'cfgfile', help = "Config file" )
    parser.add_argument("--weights", type = str, default = "checkpoints/yolov3.weights", dest = 'weightsfile', help = "Weights file" )
    parser.add_argument("--cls", type = str, default = "config/coco.names", dest = 'classes', help = "Names file" )
    parser.add_argument("--reso", type = str, default = "416", dest = 'reso', help = "Input Image Resolution" )
    
    return parser.parse_args()

# Get input arguments    
args = arg_parse()
batch_size = args.batch_size
confidence = args.confidence
nms_thresh = args.nms_thresh
classes = load_classes(args.classes)
num_classes = 80
CUDA = torch.cuda.is_available()
start = 0

# Create the network
print("Loading Network...")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded :)")
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

if CUDA:
    model.cuda()

model.eval()

################################################### Detection on Images ###################################################
images = args.images
if images:
    # Draw bounding boxes, defined locally so it can access colors list
    def draw_bboxes(x, batches, results): 
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results[int(x[0])]
        cls = int(x[-1])
        label = "{0}".format(classes[cls])
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
        
        return img

    read_dir = time.time()
    # Store input image paths in imlist
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print("No file or directory with the name {}".format(images))
        exit()
    # Create a detection results directory if it doesn't exist
    if not os.path.exists(args.detection):
        os.makedirs(args.detection)
    # Load the images
    load_batch = time.time()
    # PyTorch Variables for images
    batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
    # List containing dimensions of original images
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
    if CUDA:
        im_dim_list = im_dim_list.cuda()

    # Create input batches
    leftover = 0
    if (len(im_dim_list) % batch_size):
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover            
        im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size, len(im_batches))]))  for i in range(num_batches)]

    # Detection Loop
    write = 0
    i = 0
    obj = {}
    start_det_loop = time.time()
    for batch in im_batches:
        start = time.time()
        if CUDA:
            batch = batch.cuda()
            
        with torch.no_grad():
            prediction = model(Variable(batch))
            
        prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thresh)
        
        if type(prediction) == int:
            i += 1
            continue
        end = time.time()

        prediction[:,0] += i*batch_size # Transform batch index to imlist index

        if not write:
            output = prediction
            write = 1
        else: 
            output = torch.cat((output, prediction))
        
        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")

        i += 1

        if CUDA:
            torch.cuda.synchronize() # Sync CUDA with CPU

    try:
        output
    except NameError:
        print("No detections were made :/")
        exit()

    # Drawing Bounding boxes on images
    # Transform bounding boxes co-ordinates to account for padded images, if any
    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
    scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)
    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
    # Scale bouding boxes to accound for original dimensions
    output[:,1:5] /= scaling_factor
    # Clip bounding boxes that exceed image boundaries to edges of the image
    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])

    # Get bounding boxes on the outputs, of different colors and category name in the corner
    output_recast = time.time()
    class_load = time.time()
    colors = pkl.load(open("reference/pallete", "rb")) # Different Colors to choose from for different categories
    draw = time.time()

    # Save the bounding box drawn images in detection results directory
    list(map(lambda x: draw_bboxes(x, im_batches, orig_ims), output)) # Modify loaded_ims inplace
    det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.detection,x.split("/")[-1])) # Prefix with det_
    list(map(cv2.imwrite, det_names, orig_ims)) # Write the images in det_names
    end = time.time()

    # Print the Summary
    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
    print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
    print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
    print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("---------------------------------")
    print("{:25s}: {:2.3f}".format("Average time per image", (end - load_batch)/len(imlist)))
    print("----------------------------------------------------------")


    torch.cuda.empty_cache()


################################################### Detection on Video ###################################################
else:
    videofile = args.videofile
    camera = args.camera
    if videofile:
        cap = cv2.VideoCapture(videofile) # Video input
    else:
        cap = cv2.VideoCapture(camera) # Camera input

    assert cap.isOpened(), 'Cannot capture source'
    # Draw bounding boxes, defined locally so it can access colors list
    def draw_bboxes(x, results):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results
        cls = int(x[-1])
        color = random.choice(colors)
        label = "{0}".format(classes[cls])
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
        return img
        
    frames = 0
    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read() 
        if ret:   
            img, orig_im, dim = prep_frame(frame, inp_dim)
            # cv2.imshow("a", frame)
            im_dim = torch.FloatTensor(dim).repeat(1,2)             
            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()
            
            with torch.no_grad():
                prediction = model(Variable(img))
            
            prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thresh)
            
            if type(prediction) == int:
                frames += 1
                print("FPS of the video is {:5.4f}".format( frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
            
            prediction[:,1:5] = torch.clamp(prediction[:,1:5], 0.0, float(inp_dim))/inp_dim
            prediction[:,[1,3]] *= frame.shape[1]
            prediction[:,[2,4]] *= frame.shape[0]

            colors = pkl.load(open("reference/pallete", "rb"))

            list(map(lambda x: draw_bboxes(x, orig_im), prediction))
            
            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print(time.time() - start)
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
        else:
            break    