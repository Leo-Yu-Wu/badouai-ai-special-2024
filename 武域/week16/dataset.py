import os
import torch
import glob
import cv2
from torch.utils.data import Dataset
import random

class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # Initializing the function, read all images under data_path
        self.data_path = data_path
        self.image_path = glob.glob(os.path.join(data_path, "image/*.png"))

    def augmentation(self, image, flip):
        # Flipping using cv2.flip, flip： 1 = horizontal, 0 = vertical, -1 = both
        flip = cv2.flip(image, flip)
        return flip

    def __getitem__(self, index):
        # Get image based on index
        image_path = self.image_path[index]

        # Generate label path based on image path, simply replace 'image' with 'label' on directory
        label_path = image_path.replace('image', 'label')

        # load image and label
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)

        # convert to gray scale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        # Add a channel size, now we have [channel, height, width]
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])

        # Normalizing image
        if label.max() > 1:
            label = label / 255

        # Randomly perform flipping
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augmentation(image, flipCode)
            label = self.augmentation(label, flipCode)
        return image, label

    def __len__(self):
        return len(self.image_path)

if __name__ == '__main__':
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/train"))
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(isbi_dataset, batch_size=1, shuffle=True)
