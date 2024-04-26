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

include_dataset_csv = 'C:/Users/spsar/capstone/sample.csv'

feature_base_path='C:/Users/spsar/capstone/samples/features/transforms'
dataset_base_path='C:/Users/spsar/capstone/samples'


# feature_base_path='/content/drive/MyDrive/CapstoneProject-ISL-SignLanguageTranslation/kaggle/transforms'
# include_dataset_csv='/content/drive/MyDrive/CapstoneProject-ISL-SignLanguageTranslation/Datasets/Files-INCLUDE.csv'

include_dataset=pd.read_csv(include_dataset_csv)

files_status={}

if 'status' not in include_dataset.columns:
  include_dataset['status'] = 'Not started'

if 'total_frames' not in include_dataset.columns:
  include_dataset['total_frames'] = 0

if 'processed_frames' not in include_dataset.columns:
  include_dataset['processed_frames'] = 0

if 'percent_completion' not in include_dataset.columns:
  include_dataset['percent_completion'] = 0


def update_count(filename_contains, expression_contains, type_contains):
  k=(filename_contains, expression_contains, type_contains)
  if k in files_status:
    frames_processed=files_status[(filename_contains, expression_contains, type_contains)]
    files_status[(filename_contains, expression_contains, type_contains)]=frames_processed+1
  else:
    files_status[(filename_contains, expression_contains, type_contains)]=1


def update_dataframe(include_dataset):
  for filename_contains, expression_contains, type_contains in files_status:
    mask = (include_dataset['Filepath'].str.contains(filename_contains) &
        include_dataset['expression'].str.contains(expression_contains) &
        include_dataset['type'].str.contains(type_contains))
    
    entry=include_dataset.loc[mask]

    cap = cv2.VideoCapture(os.path.join(dataset_base_path,entry.iloc[0, 0])) 
    property_id = int(cv2.CAP_PROP_FRAME_COUNT)  
    length = int(cv2.VideoCapture.get(cap, property_id)) 
    no_of_frames_processed=files_status[(filename_contains, expression_contains, type_contains)]
    include_dataset.loc[mask,'total_frames'] = length
    include_dataset.loc[mask,'processed_frames']=no_of_frames_processed
    include_dataset.loc[mask,'status'] = 'In Progress' if length>no_of_frames_processed else 'Completed'
    include_dataset.loc[mask,'percent_completion']=f'{no_of_frames_processed/length*100 :0.1f}'
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  filename=f'{feature_base_path}/STATUS.csv'
  include_dataset.to_csv(filename, index=False)


def processJson(json_path,type_folder,expression_folder,file_name):
     with open(json_path, 'r') as json_file:
        
        try:
            data = json.load(json_file)
            # Append the data to the list
            model_type = 'body25' 
            candidate=np.array(data['candidate'])
            subset=np.array(data['subset'])
            hand_peaks=[np.array(a) for a in data['all_hand_peaks']]
            # canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)

            (bodypose_circles,bodypose_sticks,)=util.get_bodypose(candidate,subset,model_type)
            (handpose_edges,handpose_peaks)=util.get_handpose(hand_peaks)
            # canvas=drawStickmodel(canvas,bodypose_circles,bodypose_sticks,handpose_edges,handpose_peaks)
            # cv2.imwrite(f'{json_path}.stick.jpg', canvas) 
            # print(f'created {json_path}.stick.jpg')
            # canvas=util.crop_to_drawing(canvas)
            # cv2.imwrite(f'{json_path}.stick.cropped.jpg', canvas) 
            print(f'processed file {json_path}')
            feature={
                'Type': type_folder,
                'Expression': expression_folder,
                'FileName': file_name.split('-')[0],
                'Frame': int(file_name.split('-')[-1].split('.')[0]),  # Extract frame number from file name
                'Candidate': np.array(data['candidate']),
                'Subset': np.array(data['subset']),
                'AllHandPeaks':[np.array(a) for a in data['all_hand_peaks']],
                'bodypose_circles':bodypose_circles,
                'bodypose_sticks':bodypose_sticks,
                'handpose_edges':handpose_edges,
                'handpose_peaks':handpose_peaks
                }

            for idx,(body_x,body_y) in enumerate(bodypose_circles):
                feature[f'bodypeaks_x_{idx}']=body_x
                feature[f'bodypeaks_y_{idx}']=body_y

            for idx,(meanx,meany,angle,length) in enumerate(bodypose_sticks):
                feature[f'bodyedges_meanx_{idx}']=meanx
                feature[f'bodyedges_meany_{idx}']=meany
                feature[f'bodyedges_angle_{idx}']=angle
                feature[f'bodyedges_length_{idx}']=length

            for idx,hand_peaks in enumerate(handpose_peaks):
                for (hand_x, hand_y, peaktxt) in hand_peaks:
                    feature[f'hand{idx}peaks_x_{peaktxt}']=hand_x
                    feature[f'hand{idx}peaks_y_{peaktxt}']=hand_y
                    feature[f'hand{idx}peaks_peaktxt{peaktxt}']=peaktxt

            for idx,handedges in enumerate(handpose_edges):
                for (peaktxt, (handedge_x1, handedge_y1), (handedge_x2, handedge_y2)) in handedges:
                    feature[f'hand{idx}edge_x1_{peaktxt}']=handedge_x1
                    feature[f'hand{idx}edge_y1_{peaktxt}']=handedge_y1
                    feature[f'hand{idx}edge_x2_{peaktxt}']=handedge_x2
                    feature[f'hand{idx}edge_y2_{peaktxt}']=handedge_y2

            update_count(file_name.split('-')[0],expression_folder,type_folder)
        except Exception as e:
            print(f'Error while parsing JSON {json_path}')
            return {}
        return feature

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
                            for idx,file_name in enumerate(os.listdir(feature_path)):
                                if file_name.endswith('.json'):
                                    json_path = os.path.join(feature_path, file_name)                                   
                                    feature=processJson(json_path,type_folder,expression_folder,file_name)
                                    data_list.append(feature)
                                    

                                
    return data_list



if __name__ == "__main__":
    # processJson('C:/Users/spsar/capstone/samples/features/transforms/Adjectives/Blind/MVI_9585-original/MVI_4936.MOV-7.json','Adjectives','Blind','MVI_4936.MOV-7.json')
    data_list=collect_file_paths(feature_base_path)
    update_dataframe(include_dataset)
    df = pd.DataFrame(data_list)
    
    # Save the DataFrame to a CSV file
    df.to_csv('data.csv', index=False)
    print('loop')



