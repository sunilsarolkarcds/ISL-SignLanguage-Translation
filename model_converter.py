import onnx


from onnx2keras import onnx_to_keras

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

dummy_input = torch.rand(1,3,640,480)  # Example input


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# dummy_input.to(device)

torch.onnx.export(model=body_estimation_pytorch.model,args=dummy_input, f='model/bodypose_25_model.onnx')

onnx_model = onnx.load('model/bodypose_25_model.onnx')

keras_model = onnx_to_keras(onnx_model,['model0.conv1_1'])

# Save the Keras model
keras_model.save("model.h5")



print("Finished")
