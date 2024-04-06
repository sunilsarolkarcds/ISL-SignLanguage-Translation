import copy
import numpy as np
import cv2
from glob import glob
import os
import argparse
import json

# video file processing setup
# from: https://stackoverflow.com/a/61927951
import argparse
from torchinfo import summary
from src.body import Body
from src.hand import Hand
from src.model_keras import ISLSignPos


model_type = 'body25'  # 'coco'  #  
if model_type=='body25':
    model_path = './model/pose_iter_584000.caffemodel.pt'
else:
    model_path = './model/body_pose_model.pth'
body_estimation_pytorch = Body(model_path, model_type)
hand_estimation_pytorch = Hand('model/hand_pose_model.pth')

isl_sign_pos_model=ISLSignPos(pt_body_model=body_estimation_pytorch.model,pt_hand_model=hand_estimation_pytorch.model)

print("body_estimation_pytorch summary")
summary(body_estimation_pytorch.model)
print("End body_estimation_pytorch summary")

print("hand_estimation_pytorch summary")
summary(hand_estimation_pytorch.model)
print("End hand_estimation_pytorch summary")

print("isl_sign_pos_model summary")
isl_sign_pos_model.summary()
print("End isl_sign_pos_model summary")


print("Finished")
