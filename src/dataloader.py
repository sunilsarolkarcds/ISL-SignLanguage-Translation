import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_video
import os.path
from torchvision.transforms import v2
from src.ISL_Model_parameter import ISLSignPos
from src.body import Body
from src.hand import Hand
from datetime import datetime
import json
import pims
import time
import zipfile
import shutil
import copy
import src.util as util
import torch.distributed
import numpy as np
import torch.multiprocessing as mp
from torchvision.transforms.functional import to_pil_image
import datetime
import gc

include_dataset_csv='/content/drive/MyDrive/CapstoneProject-ISL-SignLanguageTranslation/Datasets/Files-INCLUDE.csv'
dataset_base_path='/content/drive/MyDrive/CapstoneProject-ISL-SignLanguageTranslation/Datasets/INCLUDE'
feature_base_path='/content/drive/MyDrive/CapstoneProject-ISL-SignLanguageTranslation/kaggle'

import os
import keras
os.environ["KERAS_BACKEND"] = "torch"
expression_type=['Days_and_Time','People','Colours','Electronics','Means_of_Transportation','Animals']



class ISLTrainerDataset(Dataset):
    def __init__(self, dataset, transforms=None):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        # self.transforms=transforms
        self.transforms = [transform.to(self.device) for transform in transforms]
        self.dataset_base_path=dataset_base_path
        self.feature_base_path=feature_base_path
        self.transforms_path_parent = os.path.join(self.feature_base_path, 'transforms')
        self.test=True
        self.model_type = 'body25'

    def isl(self,origImage):
       if self.model==None:
            if self.model_type=='body25':
                model_path = './model/pose_iter_584000.caffemodel.pt'
            else:
                model_path = './model/body_pose_model.pth'
            body_estimation_pytorch = Body(model_path, self.model_type)
            hand_estimation_pytorch = Hand('model/hand_pose_model.pth')
            # loaded_model = keras.saving.load_model("model/model3_trainaccuracy_83_valaccuracy_86.keras")
            self.model=ISLSignPos(body_estimation_pytorch.model,hand_estimation_pytorch.model)
            
       with torch.no_grad():
        return self.model(origImage)


    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        video_path = os.path.join(self.dataset_base_path,self.dataset.iloc[idx, 0])
        label_type = self.dataset.iloc[idx, 1]
        label_expression = self.dataset.iloc[idx, 2]
        print('[STARTED]',video_path,label_type,label_expression)
        features=self.extract_features_worker(video_path,label_type,label_expression)
        return features

    def ensure(self,directory_path):
      if not os.path.exists(directory_path):
        os.makedirs(directory_path)
      return directory_path

    def saveFeaturesDict(self,features,process_id,filename):
      if len(features)==0:
         return
      print(f'saving outputs for process#{process_id} {filename}')
      timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
      csv_filename = os.path.join(self.feature_base_path, f"{filename}_{timestamp}.csv")
      # print('features',features)
      df_folder = pd.DataFrame(features)
      df_folder.to_csv(csv_filename, index=False)

    def is_processed(self,filename,idx,transform, label_type, label_expression):
        transforms_path_local = os.path.join(self.transforms_path_parent, label_type, label_expression)
        directory_path = os.path.join(transforms_path_local, f"{filename.split('.')[0]}-{transform}")

        return (os.path.exists(os.path.join(directory_path, f'{filename}-{str(idx)}.json'))) and (os.path.exists(os.path.join(directory_path, f"{filename.split('.')[0]}-{str(idx)}.jpg")))

    def get_feature(self,feature):
        (bodypose_x_ytupple,bodypose_x_y_sticks)=util.get_bodypose(feature[0],feature[1],self.model_type)
        (handpose_edges,handpose_peaks)=util.get_handpose(feature[2])
        return ((bodypose_x_ytupple,bodypose_x_y_sticks),(handpose_edges,handpose_peaks))

    def extract_features_worker(self,video_path, label_type, label_expression):
        process_id=os.getpid()
        print(f'process#{process_id} processing {video_path}')
        filename = video_path.split('/')[-1]
        original_video_path = os.path.join(self.dataset_base_path, video_path)
        features = []
        video = pims.Video(original_video_path)
        totalFrames=len(video)
        start_time = time.time()
        for idx,frame in enumerate(video):
          if self.is_processed(filename,idx,'original', label_type, label_expression):
             print(f'[ALREADY PROCESSED] process#{process_id} ALREADY PROCESSED frame#{idx}/{totalFrames} of {original_video_path}')
             continue
          canvas=copy.deepcopy(frame)
          print(f'process#{process_id} processing frame#{idx}/{totalFrames} of {original_video_path}')
          model_feature = self.isl(frame[:, :, ::-1])  # Move frame to device
          ((bodypose_x_ytupple,bodypose_x_y_sticks),(handpose_edges,handpose_peaks))=self.get_feature(model_feature)
          del frame
          del canvas
          del model_feature
          gc.collect()
          torch.cuda.empty_cache()
          features.append(((bodypose_x_ytupple,bodypose_x_y_sticks),(handpose_edges,handpose_peaks)))

        # os.remove(original_video_path)
        return features

transformations = [
    # v2.RandomRotation(degrees=30),
    # v2.RandomSolarize(threshold=192.0),
    # normalise
]


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    mp.set_start_method('spawn',force=True)
    include_dataset=pd.read_csv(include_dataset_csv)
    df=include_dataset[(include_dataset['type'].isin(expression_type))]
    dataset = ISLTrainerDataset(df, transforms=transformations)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=15)  # Batch size of 1 for individual video processing

    for i, data in enumerate(dataloader):
        # filename = f'augmented_video_{i}.json'
        # save_json(data[0], filename)
        print(f"Saved augmented frames for video {data}")