import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from md.loader import MyDataset
from md.model import Model


def main(parameter):
    # define using gpu cpu
    GPU = True
    device = torch.device("cuda" if GPU else "cpu")
    # define data
    dataset = MyDataset("./data/train/inputs",transform = transforms.Compose([transforms.ToTensor()]))
    ld = DataLoader(dataset)

    # define model and method of opt
    model = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # define training
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(ld, 0):
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs =inputs.to(device).float()
            labels =labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            


if __name__ == "__main__":
    parameter = {}
    main(parameter)
