import cv2
import torch

# load model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# read image
img = cv2.imread('street.jpg')

# perform prediction
result = model(img)

# get result
result.show()
