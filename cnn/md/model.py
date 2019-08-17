import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset



class AlexNet(nn.Module):

    def __init__(self, input_size, num_classes=1000):
        super(AlexNet, self).__init__()
        self.input_size = input_size

        self.conv1 = nn.Conv2d(3,96,(11,11),stride=4)
        self.pool1 = nn.MaxPool2d(3,2)
        self.batchnorm1 = nn.BatchNorm2d(96)

        self.conv2 = nn.Conv2d(96,256,(5,5),stride=1,padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.batchnorm2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256,384,(3,3),stride=1,padding=1)
        self.conv4 = nn.Conv2d(384,384,(5,5),stride=1,padding=2)
        self.conv5 = nn.Conv2d(384,256,(3,3),stride=2,padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3,stride=2)

        self.fc1 = nn.Linear(256*3*3,1024)
        self.fc2 = nn.Linear(1024,1024)
        self.fc3 = nn.Linear(1024,2)
        self.drop = nn.Dropout()
        
        
    def forward(self,x):
        # input layer
        x_input = x

        # block 1
        x1 = F.relu(self.conv1(x_input))
        x1 = self.pool1(x1)
        x1 = self.batchnorm1(x1)
        
        # block 2
        x2 = F.relu(self.conv2(x1))
        x2 = self.pool2(x2)
        x2 = self.batchnorm2(x2)
        
        # block 3
        x3 = self.conv3(x2)
        x3 = F.relu(x3)

        # block 4
        x4 = self.conv4(x3)
        x4 = F.relu(x4)
        
        # block 5
        x5 = self.conv5(x4)
        x5 = self.pool3(F.relu(x5))
        
        # tensor to vector and full conected network
        x6 = x5.view(-1,256*3*3)
        x7 = F.relu(self.fc1(x6))
        x7 = self.drop(x7)
        x8 = F.relu(self.fc2(x7))
        x8 = self.drop(x8)
        
        # output layer
        x9 = F.softmax(self.fc3(x8))
        return x9

if __name__ == "__main__":
    from loader import MyDataset
    from torchvision import transforms
    
    dataset = MyDataset("../data/train/inputs",transform = transforms.Compose([transforms.ToTensor()]))

    ld = DataLoader(dataset)
    model = AlexNet(256,2).float()

    for i,j in ld:
        i = i.float()
        y = model(i)
        print(y.shape)
        break

