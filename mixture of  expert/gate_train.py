import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose,ToTensor 
import torch
from md.loader_gate import MyDataset
from md.model_gate import Gate_Net as Model


class Main:
    def __init__(self,parameter):
        self.parameter = parameter

        # define training data
        self.train_dataset = MyDataset(self.parameter["gate_train_data_dir"],transform = Compose([ToTensor()]))
        self.gate_train_ld = DataLoader(self.train_dataset,shuffle=True)

        # define test data
        self.test_dataset = MyDataset(self.parameter["gate_test_data_dir"],transform = Compose([ToTensor()]))
        self.gate_test_ld = DataLoader(self.test_dataset)

        #define using gpu or not
        self.GPU = True
        self.device = torch.device("cuda" if self.GPU else "cpu")

        # define model and method of opt
        self.gate_model = Model().to(self.device)
        self.criterion = nn.BCELoss()
        self.gate_optimizer = optim.SGD(self.gate_model.parameters(),lr=0.001,momentum=0.9)

    def main(self):
        self.train_gate()
    
    def train_gate(self):
        self.gate_model.train()
        for epoch in range(self.parameter["gate_epochs"]):
            print("now: {}epch".format(epoch))
            for i,(image,label) in enumerate(self.gate_train_ld):
                image = image.to(self.device).float()
                label = label.to(self.device).float()
                self.gate_optimizer.zero_grad()
                output = self.gate_model(image)
                loss = self.criterion(output,label)
                loss.backward()
                self.gate_optimizer.step()
                if (i+1)%100==0:
                    print("fineshed:{}".format(i+1))
            print(epoch)

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
    parameter = {"input_size":32,"gate_epochs":5,
                 "gate_train_data_dir":"./cifar_10/train/",
                 "gate_test_data_dir":"./data/test",
                 "training_rate":0.01}
    main = Main(parameter)
    main.main()
