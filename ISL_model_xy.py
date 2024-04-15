import copy
import numpy as np
import cv2
from glob import glob
import os
import argparse
import json
from matplotlib.figure import Figure
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# video file processing setup
# from: https://stackoverflow.com/a/61927951
import argparse
from torchinfo import summary

import src.util as util
from src.body import Body
from src.hand import Hand
from src.ISL_Model_parameter import ISLSignPos
import matplotlib.pyplot as plt


oriImg = cv2.imread('C:/Users/spsar/Downloads/MVI_5177.MOV-transformed/MVI_5177.MOV-GaussianBlur/MVI_5177.MOV-14.jpg')

### ORIGINAL
model_type = 'body25'  # 'coco'  #  
if model_type=='body25':
    model_path = './model/pose_iter_584000.caffemodel.pt'
else:
    model_path = './model/body_pose_model.pth'
body_estimation = Body(model_path, model_type)
hand_estimation = Hand('model/hand_pose_model.pth')

isl=ISLSignPos(body_estimation.model,hand_estimation.model)

(isl_candidate, isl_subset,isl_all_hand_peaks)=isl(oriImg)

######
x_ytupple,x_y_sticks=util.get_bodypose(isl_candidate,isl_subset,model_type)
# ((x1, y1),(x2, y2)),(x,y,txt)=util.get_handpose(all_hand_peaks)
(export_edges,export_peaks)=util.get_handpose(isl_all_hand_peaks)

canvas = copy.deepcopy(oriImg)


colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], 
        [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], 
        [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255,255,0], [255,255,85], [255,255,170],
            [255,255,255],[170,255,255],[85,255,255],[0,255,255]]
stickwidth=4

for idx,(mX,mY,angle,length) in enumerate(x_y_sticks):
    cur_canvas = canvas.copy()
    # print(f'new cv2.ellipse2Poly((int({mY}), int({mX})), (int({length} / 2), {stickwidth}), int({angle}), 0, 360, 1)')
    polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
    cv2.fillConvexPoly(cur_canvas, polygon, colors[idx])
    canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)



for idx,(x,y) in enumerate(x_ytupple):
    cv2.circle(canvas, (int(x), int(y)), 4, colors[idx], thickness=-1)
    

## Handpose
fig = Figure(figsize=plt.figaspect(canvas))
fig.subplots_adjust(0, 0, 1, 1)
fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
bg = FigureCanvas(fig)
ax = fig.subplots()
ax.axis('off')
ax.imshow(canvas)

edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
            [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

for (ie,(x1, y1),(x2, y2)) in export_edges:
    # print(f'new ax.plot([{x1}, {x2}], [{y1}, {y2}], color=matplotlib.colors.hsv_to_rgb([ie/float({len(edges)}), 1.0, 1.0]))')
    ax.plot([x1, x2], [y1, y2], color=matplotlib.colors.hsv_to_rgb([ie/float(len(edges)), 1.0, 1.0]))

width, height = ax.figure.get_size_inches() * ax.figure.get_dpi()

for (x,y,text) in export_peaks:
    # print(f"new ax.plot({x}, {y}, 'r.')")
    ax.plot(x, y, 'r.')



# print(f'NEW width = {width}, height={height}')
bg.draw()

canvas = np.fromstring(bg.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

####


cv2.imwrite('C:/Users/spsar/Downloads/MVI_5177.MOV-transformed/MVI_5177.MOV-GaussianBlur/MVI_5177.MOV-14-modified.jpg', canvas) 


print("Finished")
