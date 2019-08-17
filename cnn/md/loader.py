import glob
import os

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, path, transform=True):
        self.path = path
        self.files = glob.glob(self.path + "/*")
        self.transform = transform

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self,idx):
        # get file name and file path
        # file name used to create label
        # file path used to get image from volume
        file_path = self.files[idx]
        file_name = os.path.basename(file_path)
        # iamge transform to array
        image = Image.open(file_path)
        image = np.array(image,dtype=np.uint8)

        # array normalize 0-1
        image = image/255

        # transform image
        if self.transform:
            image = self.transform(image)
        # label name transform to label as a number
        # F-M: ferrite-martensite image
        # F  : ferrite iamge
        
        if file_name[0:3]=="F-M":
            label = np.array([0,1])
        else:
            label = np.array([1,0])
        
        return image,label

if __name__ == "__main__":
    dataset = MyDataset("../data/train",transform = transforms.Compose([transforms.ToTensor()]))
    ld = DataLoader(dataset)
    print(len(ld))

