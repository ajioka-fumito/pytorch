import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from md.sub_model import Conv_Conv,Down,Up


class UNet(nn.Module):

    def __init__(self,n_chanels,n_classes):
        super(UNet, self).__init__()    
        self.input_layer = Conv_Conv(n_chanels,64)
        self.down_1 = Conv_Conv(64,128)
        self.down_2 = Conv_Conv(128,256)
        self.down_3 = Conv_Conv(256,512)
        self.bottom = Conv_Conv(512,1024)
        self.up_3 = Conv_Conv(1024,512)
        self.up_2 = Conv_Conv(512,256)
        self.up_1 = Conv_Conv(256,128)
        self.output_layer = Conv_Conv(128,64)
        self.output = Conv_Conv(64,n_classes)
        self.Down = Down()
        self.Up = Up()

    def forward(self,x):
        # input
        x_input = x

        # input_layer 128
        x_0 = self.input_layer(x_input)
     
        # down 1 64
        x_d1 = self.Down(x_0)
        x_d1 = self.down_1(x_d1)
        #print(x_d1.shape,"x_d1")
        # down 2 32
        x_d2 = self.Down(x_d1)
        x_d2 = self.down_2(x_d2)
        #print(x_d2.shape,"x_d2")
        # down 3 16
        x_d3 = self.Down(x_d2)
        x_d3 = self.down_3(x_d3)
        #print(x_d3.shape,"x_d3")

        # bottom 8
        bottom = self.Down(x_d3)
        bottom = self.bottom(bottom)
        #print(bottom.shape,"bottom")

        # up 3
        x_u3 = self.Up(bottom)
        x_u3 = self.up_3(x_u3)
        x_u3 = torch.cat([x_u3,x_d3],dim=1)
        x_u3 = self.up_3(x_u3)
        #print(x_u3.shape,"x_u3")
        # up 2
        x_u2 = self.Up(x_u3)
        x_u2 = self.up_2(x_u2)
        x_u2 = torch.cat([x_u2,x_d2],dim=1)
        x_u2 = self.up_2(x_u2)
        #print(x_u2.shape,"x_u2")
        # up 1
        x_u1 = self.Up(x_u2)
        x_u1 = self.up_1(x_u1)
        x_u1 = torch.cat([x_u1,x_d1],dim=1)
        x_u1 = self.up_1(x_u1)
        #print(x_u1.shape,"x_u1")
        # output_layer
        x_out = self.Up(x_u1)
        x_out = self.output_layer(x_out)
        x_out = torch.cat([x_out,x_0],dim=1)
        x_out = self.output_layer(x_out)
        # putput
        output = self.output(x_out)

        return output
        

if __name__ == "__main__":
    from md.loader import *

    parameter = {"train_image_dir":"../data/train/image",
                 "train_label_dir":"../data/train/label",
                 "test_image_dir":"../data/test/iamge",
                 "test_label_dir":"../data/test/label",}
    from torch.utils.data import DataLoader
    from torchvision import transforms 
    paths = Make_dataset_paths(parameter)
    image_paths,label_paths = paths.generate_tarin_paths()
    print(len(image_paths))
    dataset = Mydataset(image_paths,label_paths,transform=transforms.Compose([transforms.ToTensor()]))
    ld = DataLoader(dataset)
    model = UNet(3,2).float()
    out = model(i)
    print(out.shape)
