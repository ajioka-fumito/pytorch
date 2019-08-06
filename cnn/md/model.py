import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x_input = x
        x_1 = self.pool(F.relu(self.conv1(x_input)))
        x_2 = self.pool(F.relu(self.conv2(x_1)))
        x_3 = x_2.view(-1, 16 * 5 * 5)
        x_4 = F.relu(self.fc1(x_3))
        x_5 = F.relu(self.fc2(x_4))
        x_6 = self.fc3(x_5)
        return x_6
