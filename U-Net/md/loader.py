import os.path as osp
from PIL import Image
import glob
import torch.utils.data as data
import random
import math
from torch.utils.data import Dataset

def generate_random_list(n):
    random_list = random.sample(range(n),k =n)
    return random_list

class Make_dataset_paths:
    def __init__(self,root_path,test_rate):
        self.root_path = root_path
        self.test_rate = test_rate
        self.size = self.dataset_size()
        self.iamge_paths,self.label_paths = self.generate_dataset_paths()
        
    def dataset_size(self):
        return len(glob.glob(self.root_path+"images/*"))
        
    def generate_dataset_paths(self):
        image_paths = glob.glob(self.root_path+"images/*")
        label_paths = glob.glob(self.root_path+"labels/*")
        return image_paths,label_paths

    def dataset_to_train_test(self):
        random_list = generate_random_list(self.size)
        train_num = math.floor(self.size*self.test_rate)
        train_image_paths,train_label_paths = list(),list()
        test_image_paths, test_label_paths = list(),list()
        for i in range(train_num):
            num = random_list[i]
            train_image_paths.append(self.iamge_paths[num])
            train_label_paths.append(self.label_paths[num])
        for i in range(train_num,self.size):
            num = random_list[i]
            test_image_paths.append(self.iamge_paths[num])
            test_label_paths.append(self.label_paths[num])
        return (train_image_paths,train_label_paths),(test_image_paths,test_label_paths)
"""
class MyTransform():
    def __init__():

    def __call__():
"""
class Mydataset(Dataset):
    
    def __init__(self,image_list,label_list,transform):
        self.image_list = image_list
        self.label_list = label_list
        #self.phase = phase
        self.transform = transform
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self,idx):
        # read iamge and label
        image = Image.open(self.image_list[idx])
        label = Image.open(self.label_list[idx])
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return image,label

    
if __name__ == "__main__":
    ob = Make_dataset_paths("./data/",0.1)
    print(ob.size)
    tr,te = ob.dataset_to_train_test()
    print(len(tr),len(te)