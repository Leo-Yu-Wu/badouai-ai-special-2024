import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from seaborn import heatmap

# Load model
# tried CMU-Visual and CMU-Preceptual but can't load
model = torch.hub.load('CMU-Preceptual-Computing-Lab/openpose', 'pose_resnet50', pretrained=True)
model.eval()

def preprocess(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform(image).unsqueeze(0)

image_path = "demo.jpg"
image = cv2.imread(image_path)
image_tensor = preprocess(image)

with torch.no_grad():
    output = model(image_tensor)
heatmap = output[0].cpu().numpy()
keypoints = np.argmax(heatmap, axis=0)

for i in range(keypoints.shape[0]):
    y, x = np.unravel_index(np.argmax(heatmap[i]), heatmap[i].shape)
    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

cv2.imshow('Keypoint', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
