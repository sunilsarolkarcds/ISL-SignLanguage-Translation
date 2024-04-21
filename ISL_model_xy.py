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
import torch
import pims
from torchvision.transforms.functional import to_pil_image

filepath='C:/Users/spsar/capstone/samples/features/transforms/Animals/Dog/MVI_2978-original/MVI_2978-0.jpg'

videopath='D:/4010759/Home/24. Table/MVI_9002.MP4'
cap = cv2.VideoCapture(videopath)

videoframe0=None
while(cap.isOpened()):
    ret, frame = cap.read()
    videoframe0=frame
    break
cap.release()

to_pil_image(videoframe0).save('C:/Users/spsar/OneDrive/Desktop/videoframe0.jpg')
print(f'videoframe0 {type(videoframe0)} {frame.shape}')

filename=videopath.split('/')[-1]
directory_path=os.path.dirname(videopath)
# cv2Img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
# print(f'oriImg {type(cv2Img)} {cv2Img.shape}')

# to_pil_image(cv2Img).save('C:/Users/spsar/OneDrive/Desktop/oriImg.cv2.jpg')

# with open('C:/Users/spsar/OneDrive/Desktop/oriImg.jpg.json', "w") as write:
#     json.dump(cv2Img.tolist(), write)



# video = pims.Video('C:/Users/spsar/capstone/samples/4010759\Animals/1. Dog/MVI_2978.MOV')
# to_pil_image(video[0]).save('C:/Users/spsar/OneDrive/Desktop/oriImg.pims.jpg')
# print(f'video[0] {type(video[0])} {video[0].shape}')
# with open('C:/Users/spsar/OneDrive/Desktop/oriImg-pims.jpg.jsom', "w") as write:
#     json.dump(video[0].tolist(), write)

# cv2Img_RGB = cv2.cvtColor(cv2Img, cv2.COLOR_BGR2RGB)

oriImg=videoframe0#[:, :, ::-1]
### ORIGINAL
model_type = 'body25'  # 'coco'  #  
if model_type=='body25':
    model_path = './model/pose_iter_584000.caffemodel.pt'
else:
    model_path = './model/body_pose_model.pth'
body_estimation = Body(model_path, model_type)
hand_estimation = Hand('model/hand_pose_model.pth')

isl=ISLSignPos(body_estimation.model,hand_estimation.model)


######
with torch.no_grad():
    (isl_candidate, isl_subset,isl_all_hand_peaks)=isl(oriImg)

with open(os.path.join(directory_path, f'{filename}.json'), "w") as write:
    json.dump({
        'candidate': isl_candidate.tolist(),
        'subset': isl_subset.tolist(),
        'all_hand_peaks': [peak.tolist() for peak in isl_all_hand_peaks]
    }, write)

x_ytupple,x_y_sticks=util.get_bodypose(isl_candidate,isl_subset,model_type)
# ((x1, y1),(x2, y2)),(x,y,txt)=util.get_handpose(all_hand_peaks)
(export_edges,export_peaks)=util.get_handpose(isl_all_hand_peaks)


canvas=util.drawStickmodel(oriImg,x_ytupple,x_y_sticks,export_edges,export_peaks)

# canvas = copy.deepcopy(oriImg)


# colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], 
#         [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], 
#         [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255,255,0], [255,255,85], [255,255,170],
#             [255,255,255],[170,255,255],[85,255,255],[0,255,255]]
# stickwidth=4

# for idx,(mX,mY,angle,length) in enumerate(x_y_sticks):
#     cur_canvas = canvas.copy()
#     # print(f'new cv2.ellipse2Poly((int({mY}), int({mX})), (int({length} / 2), {stickwidth}), int({angle}), 0, 360, 1)')
#     polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
#     cv2.fillConvexPoly(cur_canvas, polygon, colors[idx])
#     canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)



# for idx,(x,y) in enumerate(x_ytupple):
#     cv2.circle(canvas, (int(x), int(y)), 4, colors[idx], thickness=-1)
    

# ## Handpose
# fig = Figure(figsize=plt.figaspect(canvas))
# fig.subplots_adjust(0, 0, 1, 1)
# fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
# bg = FigureCanvas(fig)
# ax = fig.subplots()
# ax.axis('off')
# ax.imshow(canvas)

# edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
#             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

# for (ie,(x1, y1),(x2, y2)) in export_edges:
#     # print(f'new ax.plot([{x1}, {x2}], [{y1}, {y2}], color=matplotlib.colors.hsv_to_rgb([ie/float({len(edges)}), 1.0, 1.0]))')
#     ax.plot([x1, x2], [y1, y2], color=matplotlib.colors.hsv_to_rgb([ie/float(len(edges)), 1.0, 1.0]))

# width, height = ax.figure.get_size_inches() * ax.figure.get_dpi()

# for (x,y,text) in export_peaks:
#     # print(f"new ax.plot({x}, {y}, 'r.')")
#     ax.plot(x, y, 'r.')



# # print(f'NEW width = {width}, height={height}')
# bg.draw()

# canvas = np.fromstring(bg.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

# ####


cv2.imwrite('C:/Users/spsar/OneDrive/Desktop/MVI_2978-0.jpg-modified.jpg', canvas) 


print("Finished")
