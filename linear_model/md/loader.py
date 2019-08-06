import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torch import nn,tensor
from torch import from_numpy

class Mydataset(Dataset):
    def __init__(self,path):
        self.path = path
        self.data = pd.read_csv(path)
        self.x = self.data.iloc[:,2:24]
        self.t = self.data["GR"]
        
    def __len__(self):
        return len(self.t)
    
    def __getitem__(self,idx):
        dx = np.array(self.x.iloc[idx])
        dx = np.reshape(dx,(1,-1))
        dt = np.array(self.t.iloc[idx])
        dt = np.reshape(dt,(1,-1))

        return dx,dt

if __name__ == "__main__":
    dataset = Mydataset("./linear_model/data/koos234.csv")
    ld = DataLoader(dataset)
    for i,j in ld:
        j.long()
        print(i,j)


