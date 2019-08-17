import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose,ToTensor 
import torch
from md.loader import MyDataset
from md.model import AlexNet as Model


class Main:
    def __init__(self,parameter):
        self.parameter = parameter

        # define training data
        self.train_dataset = MyDataset(self.parameter["train_data_dir"],transform = Compose([ToTensor()]))
        self.train_ld = DataLoader(self.train_dataset)

        # define test data
        self.test_dataset = MyDataset(self.parameter["test_data_dir"],transform = Compose([ToTensor()]))
        self.test_ld = DataLoader(self.test_dataset)

        #define using gpu or not
        self.GPU = True
        self.device = torch.device("cuda" if self.GPU else "cpu")

        # define model and method of opt
        self.model = Model(self.parameter["input_size"]).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.model.parameters(),lr=0.001,momentum=0.9)

    def main(self):
        self.train()
        
    def train(self):
        for epoch in range(self.parameter["epochs"]):

            for i, (image,label) in enumerate(self.train_ld):
                self.model.train()
                # get the inputs; data is a list of [inputs, labels]
                image = image.to(self.device).float()
                label = label.to(self.device).float()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(image)
                loss = self.criterion(outputs, label)
                loss.backward()
                self.optimizer.step()

            print(loss)
            self.test()

    def test(self):
        cnt = 0
        miss_ferrite = 0 # フェライトをマルテンサイトと認識した数
        miss_martencite = 0 # マルテンサイトをフェライトと認識した数

        for i,(image,label) in enumerate(self.test_ld):
            self.model.eval()
            # get the inputs; data is a list of [inputs, labels]
            image = image.to(self.device).float()
            label = label.to(self.device).float()
            _,label = torch.max(label,dim=1)
            predict = self.model(image)
            _,predict = torch.max(predict,dim=1)

            if label==predict:
                cnt += 1
            elif predict==0:
                miss_martencite += 1
            else:
                miss_ferrite += 1
        test_num = i+1
        print("accuracy:",cnt/test_num)
        print(miss_martencite/test_num)
        print(miss_ferrite/test_num)
        
if __name__ == "__main__":
    parameter = {"input_size":256,"epochs":100,
                 "train_data_dir":"./data/train/inputs",
                 "test_data_dir":"./data/test/inputs",
                 "training_rate":0.01}
    main = Main(parameter)
    main.main()
