#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 20:42:37 2019

@author: nras
"""

# Main camera trap script for semi-automatic labelling
from keras.models import load_model
from matplotlib import pyplot
from os import listdir
import yolov3 as yolo

# load yolov3 model
model = load_model('model.h5')
# define the expected input shape for the model
input_w, input_h = 416, 416
# define our new photo
photo_filename = 'zebra.jpg'


# load and prepare image
image, image_w, image_h = yolo.load_image_pixels(photo_filename,
                                                 (input_w, input_h))
# make prediction
yhat = model.predict(image)
# summarize the shape of the list of arrays
print([a.shape for a in yhat])
# define the anchors
anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119],
           [10, 13, 16, 30, 33, 23]]
# define the probability threshold for detected objects
class_threshold = 0.6
boxes = list()
for i in range(len(yhat)):
    # decode the output of the network
    boxes += yolo.decode_netout(yhat[i][0], anchors[i],
                                class_threshold, input_h, input_w)
# correct the sizes of the bounding boxes for the shape of the image
yolo.correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
# suppress non-maximal boxes
yolo.do_nms(boxes, 0.5)
# define the labels
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
          "truck",	"boat", "traffic light", "fire hydrant", "stop sign",
          "parking meter", "bench",	"bird", "cat", "dog", "horse", "sheep",
          "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
          "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
          "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
          "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
          "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
          "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
          "donut", "cake", "chair", "sofa", "pottedplant", "bed",
          "diningtable", "toilet", "tvmonitor", "laptop", "mouse",	"remote",
          "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
          "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
          "hair drier", "toothbrush"]
# get the details of the detected objects
v_boxes, v_labels, v_scores = yolo.get_boxes(boxes, labels, class_threshold)
# summarize what we found
for i in range(len(v_boxes)):
    print(v_labels[i], v_scores[i])
# draw what we found
yolo.draw_boxes(photo_filename, v_boxes, v_labels, v_scores)
# The length of v_boxes tells me how many objects have been identified
# print(len(v_boxes))

# Try to cut out the image
orig_img = pyplot.imread(photo_filename)
for i in range(len(v_boxes)):
    box = v_boxes[i]
    y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
    print(y1, x1, y2, x2)
    sub_img = orig_img[y1:y2, x1:x2, 0:3]
    pyplot.imshow(sub_img)
    pyplot.show()

# Check the listdir works ok
loaded_images = list()
for filename in listdir('jpg'):
    # Load and display images
    img_data = pyplot.imread('jpg/' + filename)
    # Store loaded image
    loaded_images.append(img_data)
    print(filename)

for i in range(len(loaded_images)):
    pyplot.imshow(loaded_images[i])
    pyplot.show()
