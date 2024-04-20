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
from datetime import datetime
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
include_dataset_csv='C:/Users/spsar/capstone/sample.csv'
dataset_base_path='C:/Users/spsar/capstone/samples/'
feature_base_path='C:/Users/spsar/capstone/samples/features'

model_type = 'body25'  # 'coco'  #
if model_type=='body25':
    model_path = './model/pose_iter_584000.caffemodel.pt'
else:
    model_path = './model/body_pose_model.pth'
body_estimation_pytorch = Body(model_path, model_type)
hand_estimation_pytorch = Hand('model/hand_pose_model.pth')
model=ISLSignPos(body_estimation_pytorch.model,hand_estimation_pytorch.model)

class IncludeDatasetFeatureExtractorDataset(Dataset):
    def __init__(self, dataset, transforms=None):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        # self.transforms=transforms
        self.transforms = [transform.to(self.device) for transform in transforms]
        
        self.dataset_base_path=dataset_base_path
        self.feature_base_path=feature_base_path
        self.transforms_path_parent = os.path.join(self.feature_base_path, 'transforms')
        self.test=True

    def isl(self,origImage):
       with torch.no_grad():
        return model(origImage)
    

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        video_path = os.path.join(self.dataset_base_path,self.dataset.iloc[idx, 0])
        label_type = self.dataset.iloc[idx, 1]
        label_expression = self.dataset.iloc[idx, 2]
        print('[STARTED]',video_path,label_type,label_expression)
        features=self.extract_features_worker(video_path,label_type,label_expression)
        return features

    def zip_and_move(self,zipfilename, sourcefolderpath, destinationpath):
      with zipfile.ZipFile(zipfilename, 'w') as zipf:
          # Walk through all the files in the specified folder
          for root, dirs, files in os.walk(sourcefolderpath):
              for file in files:
                  # Create the full path to the file
                  file_path = os.path.join(root, file)

                  # Add the file to the zip archive
                  # The first argument is the file path, and the second argument is the name inside the zip file
                  # We use os.path.relpath to make sure the file paths inside the zip archive are relative to the root
                  zipf.write(file_path, os.path.relpath(file_path, sourcefolderpath))

          print(f'moving file {zipfilename} to {destinationpath}')
          shutil.move(zipfilename, destinationpath)

    def ensure(self,directory_path):
      if not os.path.exists(directory_path):
        os.makedirs(directory_path)
      return directory_path

    def saveFeaturesDict(self,features,process_id,filename):
      if len(features)==0:
         return
      print(f'saving outputs for process#{process_id} {filename}')
      timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
      csv_filename = os.path.join(self.feature_base_path, f"{filename}_{timestamp}.csv")
      # print('features',features)
      df_folder = pd.DataFrame(features)
      df_folder.to_csv(csv_filename, index=False)

    def is_processed(self,filename,idx,transform, label_type, label_expression):
        transforms_path_local = os.path.join(self.transforms_path_parent, label_type, label_expression)
        directory_path = os.path.join(transforms_path_local, f"{filename.split('.')[0]}-{transform}")

        return (os.path.exists(os.path.join(directory_path, f'{filename}-{str(idx)}.json'))) and (os.path.exists(os.path.join(directory_path, f"{filename.split('.')[0]}-{str(idx)}.jpg")))

        

    def saveFeature(self,filename, frame, idx, transform, feature, label_type, label_expression):
        transforms_path_local = os.path.join(self.transforms_path_parent, label_type, label_expression)
        directory_path = os.path.join(transforms_path_local, f"{filename.split('.')[0]}-{transform}")
        self.ensure(directory_path)
        model_type = 'body25'
        (bodypose_x_ytupple,bodypose_x_y_sticks)=util.get_bodypose(feature[0],feature[1],model_type)
        (handpose_edges,handpose_peaks)=util.get_handpose(feature[2])
        # Path(os.path.join(transforms_path_local, f'{filename.split('.')[0]}-{transform}')).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(directory_path, f'{filename}-{str(idx)}.json'), "w") as write:
            json.dump({
                'candidate': feature[0].tolist(),
                'subset': feature[1].tolist(),
                'all_hand_peaks': [peak.tolist() for peak in feature[2]]
            }, write)
        if self.test:
            # directory_path = os.path.join(transforms_path_local, f"{filename.split('.')[0]}-{transform}")
            # if not os.path.exists(directory_path):
            #   os.makedirs(directory_path)
            # to_pil_image(frame).save(os.path.join(directory_path, f"{filename.split('.')[0]}-{str(idx)}.jpg"))
            frame=util.drawStickmodel(frame,bodypose_x_ytupple,bodypose_x_y_sticks,handpose_edges,handpose_peaks)
            to_pil_image(frame).save(os.path.join(directory_path, f"{filename.split('.')[0]}-{str(idx)}.jpg"))


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

    def extract_features_worker(self,video_path, label_type, label_expression):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        process_id=os.getpid()
        print(f'process#{process_id} processing {video_path}')
        filename = video_path.split('/')[-1]
        original_video_path = os.path.join(self.dataset_base_path, video_path)
        # frames, audio, info = read_video(os.path.join(original_video_path), pts_unit='sec', output_format='TCHW')
        features = []
        video = pims.Video(original_video_path)
        totalFrames=len(video)
        start_time = time.time()
        for idx,frame in enumerate(video):
          if idx>5:
             break
          if self.is_processed(filename,idx,'original', label_type, label_expression):
             print(f'[ALREADY PROCESSED] process#{process_id} ALREADY PROCESSED frame#{idx}/{totalFrames} of {original_video_path}')
             continue
          canvas=copy.deepcopy(frame)
          # oriImg = cv2.imread('C:/Users/spsar/OneDrive/Desktop/MVI_2978-0.jpg')
          print(f'process#{process_id} processing frame#{idx}/{totalFrames} of {original_video_path}')
          model_feature = self.isl(frame[:, :, ::-1])  # Move frame to device
          #print(f'features  frame#{idx}/{totalFrames} of {original_video_path} --- {model_feature}')
          feature_entry=self.saveFeature(filename, canvas, idx, 'original', model_feature, label_type, label_expression)
          features.append(feature_entry)

        end_time = time.time()
        execution_time = end_time - start_time
        self.saveFeaturesDict(features,process_id,f'output_{process_id}_{filename}_exectime-{execution_time:.4f}')

        # os.remove(original_video_path)
        return features

transformations = [
    # v2.RandomRotation(degrees=30),
    # v2.RandomSolarize(threshold=192.0),
    # normalise
]

# transform = va.VideoAugment(seq)  # Video augmentation pipeline
# mp.set_start_method('spawn')
# Create dataset and dataloader


# model = torch.nn.parallel.DistributedDataParallel(model)
if __name__ == "__main__":
    mp.set_start_method('spawn',force=True)
    include_dataset=pd.read_csv(include_dataset_csv)
    dataset = IncludeDatasetFeatureExtractorDataset(include_dataset, transforms=transformations)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=3)  # Batch size of 1 for individual video processing

    for i, data in enumerate(dataloader):
        filename = f'augmented_video_{i}.json'
        # save_json(data[0], filename)
        print(f"Saved augmented frames for video {data} to {filename}")
