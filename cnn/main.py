import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from md.loader import MyDataset
from md.model import Model


class Main:
    def __init__(self,parameter):
        self.parameter = parameter

        # define data
        self.dataset = MyDataset(self.parameter["train_data_dir"],transform = transforms.Compose([transforms.ToTensor()]))
        self.ld = DataLoader(self.dataset)

        #define using gpu or not
        self.GPU = True
        self.device = torch.device("cuda" if self.GPU else "cpu")

        # define model and method of opt
        self.model = Model(self.parameter["input_size"]).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.ls = 0.01
        self.momentum = 0.9
        self.optimizer = optim.SGD(self.model.parameters(),lr=0.001,momentum=self.momentum)


    def main(self):
        self.train()
        self.test()
            
    def train(self):
        for epoch in range(self.parameter["epochs"]):  # loop over the dataset multiple times

            for i, data in enumerate(self.ld):
                self.model.train()
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs =inputs.to(self.device).float()
                labels =labels.to(self.device).long()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()
            print(loss)

    #def test(self):


        

if __name__ == "__main__":
    parameter = {"input_size":256,"epochs":100,
                 "train_data_dir":"./data/train/inputs",
                 "test_data_dit":"./data/test/inputs",
                 "training_rate":0.001}
    main = Main(parameter)
    main.main()
