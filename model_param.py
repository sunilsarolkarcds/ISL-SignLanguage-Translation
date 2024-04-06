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
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple
from torchinfo import summary
import src.keras.body as kerasBody
import src.keras.hand as kerasHand

from keras import Input


class FFProbeResult(NamedTuple):
    return_code: int
    json: str
    error: str


def ffprobe(file_path) -> FFProbeResult:
    command_array = ["ffprobe",
                     "-v", "quiet",
                     "-print_format", "json",
                     "-show_format",
                     "-show_streams",
                     file_path]
    result = subprocess.run(command_array, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return FFProbeResult(return_code=result.returncode,
                         json=result.stdout,
                         error=result.stderr)


# openpose setup
from src import model
from src import util
from src.body import Body
from src.hand import Hand
import torch

model_type = 'body25'  # 'coco'  #  
if model_type=='body25':
    model_path = './model/pose_iter_584000.caffemodel.pt'
else:
    model_path = './model/body_pose_model.pth'
body_estimation_pytorch = Body(model_path, model_type)
hand_estimation_pytorch = Hand('model/hand_pose_model.pth')

# body_estimation_keras = kerasBody.Body(model_path, model_type)
# hand_estimation_keras = kerasHand.Hand('model/hand_pose_model.pth')

input_shape = (3,640,480)  # Batch size, channels, height, width
input_shape_keras = Input(shape=(3,640,480))

# Create a dummy input tensor with the specified shape
dummy_input = torch.randn(input_shape)

print("Pytorch summary")
summary_body_estimation=summary(body_estimation_pytorch.model)

print('input names-')

input_names = summary_body_estimation.input_size[0]  # Assuming single input
print(f"Input Names: {input_names}")
print("End Pytorch summary")

# print("Pytorch summary")
# summary(hand_estimation_pytorch.model)
# print("End Pytorch summary")

# print("Keras hand_estimation_pytorch summary")
# print(type(hand_estimation_keras.model))
# hand_estimation_keras.model.summary(expand_nested=True)
# print("End Keras summary")

# print("Keras summary")
# print(type(body_estimation_keras.model))

# body_estimation_keras.model.summary(expand_nested=True)
# print("End Keras summary")



print("Finished")
