import glob
import numpy as np
import torch
import cv2
import os
from models.unet_model import UNet

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    net = UNet(n_channels=1, n_classes=1).to(device)
    net.load_state_dict(torch.load("./models/best_model.pth", map_location=device))
    net.eval()

    tests_paths = glob.glob("data/test/*.png")
    os.makedirs("data/res", exist_ok=True)
    for test_path in tests_paths:
        filename = os.path.basename(test_path).split('.')[0]
        save_res_path = f"data/res/{filename}_res.png"
        print(f"Processing: {test_path} -> {save_res_path}")
        img = cv2.imread(test_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        img_tensor = torch.from_numpy(img).to(device, dtype=torch.float32)

        pred = net(img_tensor)
        pred = np.array(pred.data.cpu()[0])[0]

        pred[pred >=0.5] = 255
        pred[pred < 0.5] = 0
        cv2.imwrite(save_res_path, pred)