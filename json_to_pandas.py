import cv2
import copy
import numpy as np

from src.body import Body
from src.hand import Hand
import os
import pandas as pd
import torch
from torchvision.io import read_video
from torchvision.transforms import functional as F
import os.path
from torchvision.transforms import v2
# from src.sign_pose import sign_pose
from src.ISL_Model_parameter import ISLSignPos
from src.body import Body
from src.hand import Hand
import json
from torch.utils.data import random_split
from torch.multiprocessing import spawn
import torch.multiprocessing as mp
import json
from json import JSONEncoder
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import time
import torch
from torchvision.transforms import functional as F
import os.path
from torchvision.transforms import v2
from torchvision.transforms.functional import to_pil_image
import multiprocessing
import src.util as util
import cv2
from src.util import drawStickmodel

df = pd.read_csv('C:/Users/spsar/source/repos/ISL-SignLanguage-Translation/datasets/Files-INCLUDE.csv')

feature_base_path='G:/My Drive/CapstoneProject-ISL-SignLanguageTranslation/sample/feature/transforms'
dataset_base_path='C:/Users/spsar/capstone/ISL-sign-language-recognition'


def collect_file_paths(root_directory):
    data_list = []
    for type_folder in os.listdir(root_directory):
        type_path = os.path.join(root_directory, type_folder)
        if os.path.isdir(type_path):
            for expression_folder in os.listdir(type_path):
                expression_path = os.path.join(type_path, expression_folder)
                if os.path.isdir(expression_path):
                    for file_folder in os.listdir(expression_path):
                        file_folder_path = os.path.join(expression_path, file_folder)
                        if os.path.isdir(file_folder_path):
                            feature_path = os.path.join(expression_path, file_folder)
                            for file_name in os.listdir(feature_path):
                                if file_name.endswith('.json'):
                                    json_path = os.path.join(feature_path, file_name)
                                    with open(json_path, 'r') as json_file:
                                        data = json.load(json_file)
                                        # Append the data to the list
                                        model_type = 'body25' 
                                        candidate=np.array(data['candidate'])
                                        subset=np.array(data['subset'])
                                        hand_peaks=[np.array(a) for a in data['all_hand_peaks']]
                                        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
                                        
                                        (bodypose_circles,bodypose_sticks,)=util.get_bodypose(candidate,subset,model_type)
                                        (handpose_edges,handpose_peaks)=util.get_handpose(hand_peaks)
                                        canvas=drawStickmodel(canvas,bodypose_circles,bodypose_sticks,handpose_edges,handpose_peaks)
                                        cv2.imwrite(f'{json_path}.stick.jpg', canvas) 
                                        print(f'created {json_path}.stick.jpg')
                                        # canvas=util.crop_to_drawing(canvas)
                                        # cv2.imwrite(f'{json_path}.stick.cropped.jpg', canvas) 
                                        coordinates = []
                                        canvasT=canvas.transpose(2,0,1)
                                        for c in range(canvasT.shape[0]):
                                            for y in range(canvasT.shape[1]):
                                                for x in range(canvasT.shape[2]):
                                                    if cv2.countNonZero(canvas[c,y:y+1, x:x+1]) > 0:
                                                        coordinates.append(c)
                                                        coordinates.append(y)
                                                        coordinates.append(x)
                                        
                                        data_list.append({
                                            'Type': type_folder,
                                            'Expression': expression_folder,
                                            'FileName': file_name,
                                            'Frame': int(file_name.split('-')[-1].split('.')[0]),  # Extract frame number from file name
                                            'Candidate': np.array(data['candidate']),
                                            'Subset': np.array(data['subset']),
                                            'AllHandPeaks':[np.array(a) for a in data['all_hand_peaks']],
                                            'bodypose_circles':bodypose_circles,
                                            'bodypose_sticks':bodypose_sticks,
                                            'handpose_edges':handpose_edges,
                                            'handpose_peaks':handpose_peaks,
                                            'canvas_shape':canvas.shape,
                                            'coordinates':coordinates
                                    })
                                
    return data_list



if __name__ == "__main__":
    data_list=collect_file_paths(feature_base_path)
    df = pd.DataFrame(data_list)

    # Save the DataFrame to a CSV file
    df.to_csv('data.csv', index=False)
    print('loop')



