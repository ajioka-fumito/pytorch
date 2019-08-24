import glob
import os

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
class All_Dataset(Dataset):
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
        #image = image/255

        # transform image
        if self.transform:
            image = self.transform(image)

        
        label = np.zeros(10)
        ans = int(file_name[0])
        label[ans] = 1
        return image,label

class Artificial_Dataset(Dataset):
    def __init__(self, path, transform=True):
        self.path = path
        self.files = glob.glob(self.path + "/*")
        self.files_natural,self.files_artificial = self.split_files()
        self.transform = transform

    def split_files(self):
        files_natural,files_artificial = [],[]

        for path in self.files:
            name = os.path.basename(path)
            if int(name[0]) in [0,1,8,9]:
                files_artificial.append(path)
            else:
                files_natural.append(path)
        return files_natural,files_artificial

    def __len__(self):
        return len(self.files_artificial)
    
    def __getitem__(self,idx):
        # get file name and file path
        # file name used to create label
        # file path used to get image from volume
        file_path = self.files_artificial[idx]
        file_name = os.path.basename(file_path)
        # iamge transform to array
        image = Image.open(file_path)
        image = np.array(image,dtype=np.uint8)

        # array normalize 0-1
        image = image/255

        # transform image
        if self.transform:
            image = self.transform(image)

        
        label = np.zeros(10)
        label[int(file_name[0])] = 1
        
        return image,label

class Natural_Dataset(Dataset):
    def __init__(self, path, transform=True):
        self.path = path
        self.files = glob.glob(self.path + "/*")
        self.files_natural,self.files_artificial = self.split_files()
        self.transform = transform

    def split_files(self):
        files_natural,files_artificial = [],[]

        for path in self.files:
            name = os.path.basename(path)
            if int(name[0]) in [0,1,8,9]:
                files_artificial.append(path)
            else:
                files_natural.append(path)
        return files_natural,files_artificial

    def __len__(self):
        return len(self.files_natural)
    
    def __getitem__(self,idx):
        # get file name and file path
        # file name used to create label
        # file path used to get image from volume
        file_path = self.files_natural[idx]
        file_name = os.path.basename(file_path)
        # iamge transform to array
        image = Image.open(file_path)
        image = np.array(image,dtype=np.uint8)

        # array normalize 0-1
        image = image/255

        # transform image
        if self.transform:
            image = self.transform(image)

        
        label = np.zeros(10)
        label[int(file_name[0])] = 1
        
        return image,label

if __name__ == "__main__":
    dataset = Artificial_Dataset("./cifar_10/train",transform = transforms.Compose([transforms.ToTensor()]))
    ld = DataLoader(dataset)
    print(len(ld))

    for i,(image,label) in enumerate(ld):
        print(label)
        if i == 10:
            break