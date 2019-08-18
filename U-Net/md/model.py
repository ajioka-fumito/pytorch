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

    def forward(self,x):
        # input
        x_input = x
        # input_layer
        x_0 = self.input_layer(x_input)
        # down 1
        x_d1 = Down(x_0)
        x_d1 = self.down_1(x_d1)
        # down 2
        x_d2 = Down(x_d1)
        x_d2 = self.down_2(x_d1)
        # down 3
        x_d3 = Down(x_d2)
        x_d3 = self.down_3(x_d2)
        # bottom
        bottom = Down(x_d3)
        bottom = self.bottom(bottom)
        # up 3
        x_u3 = Up(bottom)
        x_u3 = torch.cat([x_u3,x_d3],dim=1)
        x_u3 = self.up_3(x_u3)
        # up 2
        x_u2 = Up(x_u3)
        x_u2 = torch.cat([x_u2,x_d2],dim=1)
        x_u2 = self.up_2(x_u2)
        # up 1
        x_u1 = Up(x_u2)
        x_u1 = torch.cat([x_u1,x_d1],dim=1)
        x_u1 = self.up_1(x_u1)
        # output_layer
        x_out = Up(x_u1)
        x_out = torch.cat([x_out,x_0],dim=1)
        x_out = self.output_layer(x_out)
        # putput
        output = self.output(x_out)

        return output
        

if __name__ == "__main__":
    from loader import MyDataset
    from torchvision import transforms


