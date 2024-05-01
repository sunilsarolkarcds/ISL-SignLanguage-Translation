import copy
import numpy as np
import cv2
from glob import glob
import os
import argparse
import json
from src.ISL_Model_parameter import ISLSignPosTranslator
# video file processing setup
# from: https://stackoverflow.com/a/61927951
import argparse
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import pims
from keras.models import Sequential
import os
from keras.models import Sequential
import pickle
import keras
from keras.models import Sequential
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout,Input,BatchNormalization

from src.expression_mapping import expression_mapping
import cv2

from src.model import handpose_model, bodypose_25_model


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


translation_model = Sequential()
translation_model.add(Input(shape=((20, 156))))
translation_model.add(keras.layers.Masking(mask_value=0.))
translation_model.add(BatchNormalization())
translation_model.add(Bidirectional(LSTM(32, recurrent_dropout=0.2, return_sequences=True)))

translation_model.add(Dropout(0.2))
translation_model.add(Bidirectional(LSTM(32, recurrent_dropout=0.2)))

translation_model.add(keras.layers.Activation('elu'))
translation_model.add(Dense(32, use_bias=False, kernel_initializer='he_normal'))

translation_model.add(BatchNormalization())
translation_model.add(Dropout(0.2))
translation_model.add(keras.layers.Activation('elu'))
translation_model.add(Dense(32, kernel_initializer='he_normal',use_bias=False))

translation_model.add(BatchNormalization())
translation_model.add(keras.layers.Activation('elu'))
translation_model.add(Dropout(0.2))
translation_model.add(Dense(len(list(expression_mapping.keys())), activation='softmax'))




# def process_frame(frame):
#     # canvas = copy.deepcopy(frame)
    
#     return isl_translator(frame)

# writing video with ffmpeg because cv2 writer failed
# https://stackoverflow.com/questions/61036822/opencv-videowriter-produces-cant-find-starting-number-error
import ffmpeg

# open specified video
# parser = argparse.ArgumentParser(
#         description="Process a video annotating poses detected.")
# parser.add_argument('file', type=str, help='Video file location to process.')
# parser.add_argument('--no_hands', action='store_true', help='No hand pose')
# parser.add_argument('--no_body', action='store_true', help='No body pose')
# args = parser.parse_args()
video_file = "C:/Users/spsar/capstone/ISL-sign-language-recognition/4010759/People/81. Friend/MVI_3975.MOV"#args.file


# get video file info
ffprobe_result = ffprobe(video_file)
info = json.loads(ffprobe_result.json)
videoinfo = [i for i in info["streams"] if i["codec_type"] == "video"][0]
input_fps = videoinfo["avg_frame_rate"]
# input_fps = float(input_fps[0])/float(input_fps[1])
input_pix_fmt = videoinfo["pix_fmt"]
input_vcodec = videoinfo["codec_name"]

# define a writer object to write to a movidified file
postfix = info["format"]["format_name"].split(",")[0]
output_file = ".".join(video_file.split(".")[:-1])+".processed." + postfix

# isl_translator=ISLSignPosTranslator(body_estimation.model,hand_estimation.model,translation_model,input_fps, input_pix_fmt,
#                         input_vcodec)
isl_translator=ISLSignPosTranslator(bodypose_25_model(),handpose_model(), translation_model)
isl_translator.load_weights('model/isl-translate-v1.keras')
# isl_translator.save('model/isl-translate-v1.keras')
# isl_translator.save('path/to/location.keras') 

# video = pims.Video(video_file)

window_size=20

window=[]
cap = cv2.VideoCapture(video_file,)
totalFrames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for idx in range(totalFrames):#enumerate(file_df.rolling(window=20, step=20,min_periods=1)):
    print(f'captured frame#{idx}')
    if(cap.isOpened()):
        ret, frame = cap.read()


    if len(window)<window_size:
        window.append(frame)
    else:
        window[:-1] = window[1:]
        window[-1]=frame

        encoded_translation = isl_translator(np.array(window))
        encoded_translation=encoded_translation[0].cpu().detach().numpy()
        sorted_index=np.argsort(encoded_translation)[::-1]
        maxindex=np.argmax(encoded_translation)
        print(f'{idx} {encoded_translation[maxindex]:0.4f} {maxindex}-{expression_mapping[maxindex]} ')#{[(pi,encoded_translation[pi],expression_mapping[pi]) for pi in sorted_index]}


cap.release()



