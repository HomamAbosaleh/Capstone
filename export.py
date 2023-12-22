#import torch
from ultralytics import YOLO

# Set the device to GPU if available
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load a model
model = YOLO('./yolo8s.pt')

# Export the model
model.export(format="onnx", simplify=True, dynamic=True)

