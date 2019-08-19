import os.path as osp
from PIL import Image
import glob
import torch.utils.data as data
from torch.utils.data import Dataset
import torch
def OneHot(label):
    label = torch.squeeze(label)
    label = torch.squeeze(label)
    label = torch.eye(3)[label]
    label = label.permute(2,0,1)
    return label
class Make_dataset_paths:
    def __init__(self,parameter):
        self.parameter = parameter

    def generate_tarin_paths(self):
        image_paths = glob.glob(self.parameter["train_image_dir"]+"/*")
        label_paths = glob.glob(self.parameter["train_label_dir"]+"/*")
        return image_paths,label_paths

    def generate_test_paths(self):
        image_paths = glob.glob(self.parameter["test_image_dir"]+"/*")
        label_paths = glob.glob(self.parameter["test_label_dir"]+"/*") 
        return image_paths,label_paths

class MyDataset(Dataset):
    def __init__(self,image_list,label_list,transform):
        self.image_list = image_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self,idx):
        # read iamge and label
        image = Image.open(self.image_list[idx])
        label = Image.open(self.label_list[idx])
        if self.transform:
            image = self.transform(image)
            label = self.transform(label).long()
            label = OneHot(label)
        return image,label

    
if __name__ == "__main__":
    parameter = {"train_image_dir":"./data/train/image",
                 "train_label_dir":"./data/train/label",
                 "test_image_dir":"./data/test/iamge",
                 "test_label_dir":"./data/test/label",}
    from torch.utils.data import DataLoader
    from torchvision import transforms 
    paths = Make_dataset_paths(parameter)
    image_paths,label_paths = paths.generate_tarin_paths()
    print(len(image_paths))
    dataset = MyDataset(image_paths,label_paths,transform=transforms.Compose([transforms.ToTensor()]))
    ld = DataLoader(dataset)
    for i,j in ld:
        print(i.shape)
        print(j.shape)
        break