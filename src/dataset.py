import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os 
from PIL import Image
from glob import glob 
import numpy as np

class HumanSegmentation(Dataset):
    def __init__(self, img_transforms = None, mask_transforms = None) :
        super(HumanSegmentation,self).__init__()
        self.imgs = glob("ConferenceVideoSegmentationDataset/original_training/*")
        self.masks = [i.replace('original_training', 'ground_truth_training').replace('original', 'gt')
                     for i in self.imgs]
        self.img_transforms = img_transforms
        self.mask_transforms = mask_transforms


    def __len__(self):
        return len(self.imgs)
    

    def __getitem__(self, index):

        img = np.array(Image.open(self.imgs[index]).convert("RGB"))
        mask = np.array(Image.open(self.masks[index]).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.img_transforms:
            img = self.img_transforms(img)
        if self.mask_transforms:
            mask = self.mask_transforms(mask)

        return img, mask
