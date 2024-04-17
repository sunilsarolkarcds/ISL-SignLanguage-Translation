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
import datetime
from src.util import get_bodypose
from src.util import get_handpose
import pims



feature_base_path='C:/Users/spsar/capstone/samples/features'
dataset_base_path='C:/Users/spsar/capstone/samples/'

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)



transforms_path_parent = os.path.join(feature_base_path, 'transforms')
test = True

def ensureDirectoryExists(directory_path):
    if not os.path.exists(directory_path):
      os.makedirs(directory_path)

def saveFeature(filename, frame, idx, transform, feature, label_type, label_expression,device):
    print("saveFeature")
    transforms_path_local = os.path.join(transforms_path_parent, label_type, label_expression)
    directory_path = os.path.join(transforms_path_local, f"{filename.split('.')[0]}-{transform}")
    ensureDirectoryExists(directory_path)
    # Path(os.path.join(transforms_path_local, f'{filename.split('.')[0]}-{transform}')).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(directory_path, f'{filename}-{str(idx)}.json'), "w") as write:
        json.dump({
            'candidate': feature[0].tolist(),
            'subset': feature[1].tolist(),
            'all_hand_peaks': [peak.tolist() for peak in feature[2]]
        }, write)
    if test:
        # directory_path = os.path.join(transforms_path_local, f"{filename.split('.')[0]}-{transform}")
        # if not os.path.exists(directory_path):
        #   os.makedirs(directory_path)
        print('frame.shape',frame.shape)
        to_pil_image(frame).save(os.path.join(directory_path, f"{filename.split('.')[0]}-{str(idx)}.jpg"))

    model_type = 'body25'
    (bodypose_x_ytupple,bodypose_x_y_sticks)=get_bodypose(feature[0],feature[1],model_type)
    (handpose_edges,handpose_peaks)=get_handpose(feature[2])
    features={}
    features['transform']=transform
    features['filepath']=os.path.join(directory_path, f'{filename}-{str(idx)}.json')
    features['frame_no']=idx
    features['type']=label_type
    features['expression']=label_expression
    features['candidate']= feature[0].tolist()
    features['subset']=feature[1].tolist()
    features['all_hand_peaks']=[peak.tolist() for peak in feature[2]]
    features['bodypose_x_ytupple']=bodypose_x_ytupple
    features['bodypose_x_y_sticks']=bodypose_x_y_sticks
    features['handpose_edges']=handpose_edges
    features['handpose_peaks']=handpose_peaks
    return features

def extract_features_worker(video_path, label_type, label_expression, model, device):
    print("extract_features_worker")
    filename = video_path.split('/')[-1]
    original_video_path = os.path.join(dataset_base_path, video_path)
    print(original_video_path, filename)
    # frames, audio, info = read_video(os.path.join(original_video_path), pts_unit='sec', output_format='TCHW')
    features = []
    transformations = [
    v2.RandomRotation(degrees=30),
    v2.RandomSolarize(threshold=192.0),
    ]
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transforms = [transform.to(device) for transform in transformations]
    with torch.no_grad():
        print(f'capturing frames for {original_video_path}')
        # cap = cv2.VideoCapture(original_video_path)
        video = pims.Video(original_video_path)

        for idx,frame in enumerate(video):
          print(f'processing frame#{idx} of {original_video_path}')
          # print('######################',frame.shape)
          # orig_pil_img = F.to_pil_image(frame)
          # orig_tensor= F.to_tensor(orig_pil_img)
          # frame1 = orig_tensor.to(device)
          print('frame1.shape',frame.shape)
          print('type(frame)',type(frame))
          feature = model(frame)  # Move frame to device
          feature_entry=saveFeature(filename, frame, idx, 'original', feature, label_type, label_expression,device)
          features.extend(feature_entry)
          # del frame1
          # pil_image = F.to_pil_image(frame)
          for transformation in transformations:
              transformed_frame = transformation(frame)
              # transformed_tensor = F.to_tensor(transformed_frame)
              # transformed_tensor = transformed_tensor.to(device)  # Move transformed frame to device
              print('transformed_frame.shape',transformed_frame.shape)
              print('type(transformed_frame)',type(transformed_frame))
              transformed_feature = model(transformed_frame)
              feature_entry=saveFeature(filename, transformed_frame, idx, transformation.__class__.__name__, transformed_feature, label_type, label_expression,device)
              features.extend(feature_entry)
              # del transformed_frame
              features.extend(feature_entry)
          idx += 1
    # os.remove(original_video_path)
    return features

def extract_features_worker_wrapper(row_data,device,model):
  print("extract_features_worker_wrapper")
  model = model.to(device)
  return extract_features_worker(row_data['Filepath'], row_data['type'], row_data['expression'], model, device)


def extract_features(queue,video_dataset, num_workers,device):
  print("extract_features")
  model_type = 'body25'  # 'coco'  #
  if model_type == 'body25':
      model_path = './model/pose_iter_584000.caffemodel.pt'
  else:
      model_path = './model/body_pose_model.pth'
  body_estimation_pytorch = Body(model_path, model_type)
  hand_estimation_pytorch = Hand('model/hand_pose_model.pth')
  model = ISLSignPos(body_estimation_pytorch.model, hand_estimation_pytorch.model)
  

  features=[]

  for idx, row in video_dataset.iterrows():
    feature=extract_features_worker_wrapper(row,device,model)
    features.extend(feature)

  queue.put(features)
  # pool = ThreadPool(num_workers)
  # results = pool.map(queue,extract_features_worker_wrapper, video_dataset.iterrows())
  # pool.close()
  # pool.join()
  # return results

if __name__ == "__main__":
  mp.set_start_method('spawn',force=True)
  manager = mp.Manager()
  processing_status = manager.list()
  # multiprocessing.freeze_support()
  # model_type = 'body25'  # 'coco'  #
  # if model_type == 'body25':
  #     model_path = './model/pose_iter_584000.caffemodel.pt'
  # else:
  #     model_path = './model/body_pose_model.pth'
  # body_estimation_pytorch = Body(model_path, model_type)
  # hand_estimation_pytorch = Hand('model/hand_pose_model.pth')
  # isl = ISLSignPos(body_estimation_pytorch.model, hand_estimation_pytorch.model)
  include_dataset = pd.read_csv('C:/Users/spsar/capstone/sample.csv')

  num_workers = 2
  split_size = len(include_dataset) // num_workers
  # Define data splits for each process (can be more sophisticated)
  video_splits = [include_dataset.iloc[i:i + split_size] for i in range(0, len(include_dataset), split_size)]
  
  # Create a queue to store results
  queue = mp.Queue()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  


  processes = []
  for videos in video_splits:
    p = mp.Process(target=extract_features, args=(queue, videos,num_workers,device))
    p.start()
    print(f'started process {p}')
    processes.append(p)



  # Wait for all processes to finish
  for p in processes:
    p.join()


  # Collect results from all processes
  features_dict = []
  while not queue.empty():
    try:
      video_path, features = queue.get()
      features_dict.extend(features)
    except Exception as e:
      print(f"Error collecting result: {e}")


  timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  csv_filename = os.path.join(feature_base_path, f"output_{timestamp}.csv")
  print('features',features_dict)
  df_folder = pd.DataFrame(features_dict)
  df_folder.to_csv(csv_filename, index=False)