import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self,input_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.x_1_shape = (self.input_size-4)//2
        self.x_2_shape = (self.x_1_shape-4)//2

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(self.x_2_shape**2*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x_input = x
        x_1 = self.pool(F.relu(self.conv1(x_input)))
        print(x_1.shape)
        x_2 = self.pool(F.relu(self.conv2(x_1)))
        x_3 = x_2.view(-1,self.x_2_shape**2*16)
        x_4 = F.relu(self.fc1(x_3))
        x_5 = F.relu(self.fc2(x_4))
        x_6 = self.fc3(x_5)
        return x_6

if __name__ == "__main__":
    from loader import MyDataset
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    dataset = MyDataset("./data/train/inputs",transform = transforms.Compose([transforms.ToTensor()]))
    ld = DataLoader(dataset)

    model = Model(input_size=256).float().to("cuda")
    for i,j in ld:
        i = i.float().to("cuda")
        j = j.float().to("cuda")

        k = model(i)
