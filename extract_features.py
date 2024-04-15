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

df = pd.read_csv('C:/Users/spsar/source/repos/ISL-SignLanguage-Translation/datasets/Files-INCLUDE.csv')
include_dataset=df[df['expression'].isin(['loud','quite','happy'])]
feature_base_path='C:/Users/spsar/capstone/ISL-sign-language-recognition/features'
dataset_base_path='C:/Users/spsar/capstone/ISL-sign-language-recognition'

manager = mp.Manager()
processing_status = manager.list()

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class MyLayer(torch.nn.Module):
    def __init__(self, a, b):
        super(MyLayer, self).__init__()

    def my_function(self, x):
        return (np.array([]),np.array([]),np.array([]))

    def forward(self, x):
        return self.my_function(x)

transformations = [
    v2.RandomRotation(degrees=30),
    v2.RandomSolarize(threshold=192.0),
]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# self.transforms=transforms
transforms = [transform.to(device) for transform in transformations]


transforms_path_parent=os.path.join(feature_base_path,'transforms')
test=True

def saveFeature(filename,frame,idx,transform,feature,label_type,label_expression):
  transforms_path_local=os.path.join(transforms_path_parent,label_type,label_expression)
  (candidate, subset,all_hand_peaks)=feature
  Path(os.path.join(transforms_path_local,f'{filename}-{transform}')).mkdir(parents=True, exist_ok=True)
  with open( os.path.join(transforms_path_local,f'{filename}-{transform}',f'{filename}-{str(idx)}.json') , "w" ) as write:
    json.dump({
        'candidate':candidate.tolist(),
        'subset':subset.tolist(),
        'all_hand_peaks':[peak.tolist() for peak in all_hand_peaks]

    } , write )
  if test:
    # print("frame.shape",frame.shape)
    canvas = copy.deepcopy(frame.permute(1, 2, 0).cpu().numpy())
    # print("canvas.shape",canvas.shape)
    # canvas = util.draw_bodypose(canvas, candidate, subset, model_type)
    # canvas = util.draw_handpose(canvas, all_hand_peaks)
    # logger.info(f'saving canvas file {filename} to local machine')
    #Path(os.path.join(transforms_path_local,f'{filename}-{transform}')).mkdir(parents=True, exist_ok=True)
    to_pil_image(frame).save(os.path.join(transforms_path_local,f'{filename}',f'{filename}-{str(idx)}.jpg'))
    # logger.info(f'DONE saving canvas file {filename} to local machine')

# Function to extract features from a single video
def extract_features_worker(video_path,label_type,label_expression, model, device):
  filename=video_path.split('/')[-1]
  original_video_path=os.path.join(dataset_base_path,video_path)
  # Path(original_video_path).mkdir(parents=True, exist_ok=True)
  # shutil.copy(video_path, original_video_path)

  # Load video using OpenCV or other library
#   frames, _ = cv2.VideoCapture(original_video_path).read()

  frames, audio, info = read_video(os.path.join(original_video_path,filename), pts_unit='sec',output_format='TCHW')

  features = {}

  # Extract features from each frame
  with torch.no_grad():
      for idx,frame in enumerate(frames):
          feature = model(frame)
          features['original']=feature.cpu()
          saveFeature(filename,frame,idx,'original',feature,label_type,label_expression)
          for transformation in transformations:
            transformed_frame = transformation.ToTensor()(frame)
            transformed_feature = model(transformed_frame)
            features[transformation.__class__.__name__]=feature.cpu()
            saveFeature(filename,frame,idx,transformation.__class__.__name__,transformed_feature)

  os.remove(original_video_path)
  # Return the list of features
  return features


# Function to spawn worker processes for feature extraction
def extract_features(video_dataset, model, device, num_workers):
    # Split dataset into chunks for parallel processing
    # chunks = [video_dataset for _ in range(num_workers)]
    chunks = [df.iloc[i:i + num_workers] for i in range(0, len(df), num_workers)]
    # Spawn worker processes
    queue = mp.Queue()
    results = []
    print('spawn(')
    spawn(
        extract_features_worker_wrapper,
        args=(chunks, model, device,queue),
        nprocs=num_workers
    )

    for _ in range(num_workers):
        result = queue.get()  # Get the result from the queue
        results.append(result)

    return results

# Wrapper function for spawn to handle multiple chunks per worker
def extract_features_worker_wrapper(chunks, model, device, queue):
    cumulative_features = []
    print('extract_features_worker_wrapper(chunks, model, device, queue)')
    for chunk in chunks:
        for idx,row in chunk.iterrows():
            processing_status.append(f"Processing: {row['Filepath']}")
            signpose=extract_features_worker(row['Filepath'],row['type'],row['expression'], model.to(device), device)
            features={}
            features['filepath']=row['Filepath']
            features['type']=row['type']
            features['expression']=row['expression']
            features['signpose']=signpose
            cumulative_features.append(features)
            processing_status.append(f"Finished: {row['Filepath']}")

    queue.put(cumulative_features)
    return cumulative_features

if __name__ == "__main__":
    multiprocessing.freeze_support()
    print('loop')
    model_type = 'body25'  # 'coco'  #
    if model_type=='body25':
        model_path = './model/pose_iter_584000.caffemodel.pt'
    else:
        model_path = './model/body_pose_model.pth'
    body_estimation_pytorch = Body(model_path, model_type)
    hand_estimation_pytorch = Hand('model/hand_pose_model.pth')
    # isl=ISLSignPos(body_estimation_pytorch.model,hand_estimation_pytorch.model)
    isl=ISLSignPos(body_estimation_pytorch.model,hand_estimation_pytorch.model)
    # Choose device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Number of worker processes
    num_workers = 6
    print('extract_features(include_dataset, mylayer, device, num_workers)')
    # Extract features
    # mylayer=MyLayer(body_estimation_pytorch.model,hand_estimation_pytorch.model)
    features = extract_features(include_dataset, isl, device, num_workers)
    video_paths = include_dataset['Filepath'].tolist()

    while any(status == "Processing: " + video_path for video_path, status in zip(video_paths, processing_status)):
      # Print processing status with progress information (optional)
      print(f"Processing videos: {[status for status in processing_status if status.startswith('Processing: ')]}")
      time.sleep(5)  # Adjust sleep time for status update frequency



    # original_video_path=os.path.join(video_path,'original')
    # Path(original_video_path).mkdir(parents=True, exist_ok=True)
    # Specify the path and filename for the CSV file
    csv_filename = os.path.join(feature_base_path,"output.csv")
    df=pd.DataFrame(features)
    df.to_csv(csv_filename, index=False)
    shutil.copy(csv_filename, '/content/drive/MyDrive/CapstoneProject-ISL-SignLanguageTranslation/features')

    consolidated_features_json=json.dumps(features, cls=NumpyArrayEncoder)
    with open( os.path.join(feature_base_path,f'include_dataset_features.json') , "w" ) as write:
      write.write(consolidated_features_json)