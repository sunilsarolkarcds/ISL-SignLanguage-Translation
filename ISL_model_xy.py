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
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
print(keras.__version__)

filepath='C:/Users/spsar/capstone/samples/features/transforms/Animals/Dog/MVI_2978-original/MVI_2978-0.jpg'

videopath='C:/Users/spsar/capstone/ISL-sign-language-recognition/4010759/Adjectives/9. Nice/MVI_9590.MOV'
cap = cv2.VideoCapture(videopath)

videoframe0=None
while(cap.isOpened()):
    ret, frame = cap.read()
    videoframe0=frame
    break
cap.release()
oriImg=videoframe0
# to_pil_image(videoframe0).save('C:/Users/spsar/OneDrive/Desktop/videoframe0.jpg')

# print(f'videoframe0 {type(videoframe0)} {frame.shape}')

filename=videopath.split('/')[-1]
directory_path=os.path.dirname(videopath)
# cv2Img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
# print(f'oriImg {type(cv2Img)} {cv2Img.shape}')

# to_pil_image(cv2Img).save('C:/Users/spsar/OneDrive/Desktop/oriImg.cv2.jpg')

# with open('C:/Users/spsar/OneDrive/Desktop/oriImg.jpg.json', "w") as write:
#     json.dump(cv2Img.tolist(), write)



# video = pims.Video(videopath)
# to_pil_image(video[0]).save('C:/Users/spsar/OneDrive/Desktop/test/oriImg.pims.jpg')
# print(f'video[0] {type(video[0])} {video[0].shape}')
# with open('C:/Users/spsar/OneDrive/Desktop/test/oriImg-pims.jpg.jsom', "w") as write:
#     json.dump(video[0].tolist(), write)

# # cv2Img_RGB = cv2.cvtColor(cv2Img, cv2.COLOR_BGR2RGB)

# oriImg=video[0][:, :, ::-1]
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
def populate_features(bodypose_circles,handpose_peaks):
        # X_body_test = [f'bodypeaks_x_{i}' for i in range(15)] + [f'bodypeaks_y_{i}' for i in range(15)]
        # X_hand0_test = [f'hand0peaks_x_{i}' for i in range(21)] + [f'hand0peaks_y_{i}' for i in range(21)] + [f'hand0peaks_peaktxt{i}' for i in range(21)]
        # X_hand1_test = [f'hand1peaks_x_{i}' for i in range(21)] + [f'hand1peaks_y_{i}' for i in range(21)] + [f'hand1peaks_peaktxt{i}' for i in range(21)]

        # feature_columns_new = X_body_test + X_hand0_test + X_hand1_test
        feature=[]
        for idx in range(15):
            if(idx<len(bodypose_circles)):
                feature.append(bodypose_circles[idx][0])
            else:
                feature.append(0)
        
        for idx in range(15):
            if(idx<len(bodypose_circles)):
                feature.append(bodypose_circles[idx][1])
            else:
                feature.append(0)

        for hand_idx in range(2):
            for idx in range(21):
                if(idx<len(handpose_peaks[hand_idx])):
                    feature.append(float(handpose_peaks[hand_idx][idx][0]))
                else:
                    feature.append(0)

            for idx in range(21):
                if(idx<len(handpose_peaks[hand_idx])):
                    feature.append(float(handpose_peaks[hand_idx][idx][1]))
                else:
                    feature.append(0)

            for idx in range(21):
                if(idx<len(handpose_peaks[hand_idx])):
                    feature.append(float(handpose_peaks[hand_idx][idx][2]))
                else:
                    feature.append(0)

        # for idx in range(21):
        #     if(idx<len(handpose_peaks[1])):
        #         feature.append(handpose_peaks[1][idx][0])
        #     else:
        #         feature.append(0)
        
        # for idx in range(21):
        #     if(idx<len(handpose_peaks[1])):
        #         feature.append(handpose_peaks[1][idx][1])
        #     else:
        #         feature.append(0)

        # for idx in range(21):
        #     if(idx<len(handpose_peaks[1])):
        #         feature.append(handpose_peaks[1][idx][2])
        #     else:
        #         feature.append(0)

        # for idx,handedges in enumerate(handpose_edges):
        #     for (peaktxt, (handedge_x1, handedge_y1), (handedge_x2, handedge_y2)) in handedges:
        #         feature[f'hand{idx}edge_x1_{peaktxt}']=handedge_x1
        #         feature[f'hand{idx}edge_y1_{peaktxt}']=handedge_y1
        #         feature[f'hand{idx}edge_x2_{peaktxt}']=handedge_x2
        #         feature[f'hand{idx}edge_y2_{peaktxt}']=handedge_y2

        X=np.array(feature)
        # time_steps = 12  # Number of time steps
        # num_features = X.shape[0] // time_steps  # Number of features per time step
        # X_reshaped = X.reshape(1,time_steps, num_features)
        return X


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

features=populate_features(x_ytupple,export_peaks)
np.savetxt('C:/Users/spsar/OneDrive/Desktop/test/MVI_9590.MOV.frame1.cv2.numpy',features)


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


cv2.imwrite('C:/Users/spsar/OneDrive/Desktop/test/MVI_2978-0.jpg-modified.jpg', canvas) 


print("Finished")
