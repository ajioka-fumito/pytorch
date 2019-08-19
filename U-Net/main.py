import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose,ToTensor 
import torch
from md.loader import MyDataset,Make_dataset_paths
from md.model import UNet as Model
from md.loss import DiceLoss

class Main:
    def __init__(self,parameter):
        self.parameter = parameter

        # define training data
        self.paths = Make_dataset_paths(self.parameter)
        self.train_image_paths,self.train_label_paths = self.paths.generate_tarin_paths()
        self.train_dataset = MyDataset(self.train_image_paths,self.train_label_paths,transform = Compose([ToTensor()]))
        self.train_ld = DataLoader(self.train_dataset)

        # define test data
        """
        self.test_dataset = MyDataset(self.parameter["test_data_dir"],transform = Compose([ToTensor()]))
        self.test_ld = DataLoader(self.test_dataset)
        """
        #define using gpu or not
        self.GPU = True
        self.device = torch.device("cuda" if self.GPU else "cpu")

        # define model and method of opt
        self.model = Model(self.parameter["n_chanels"],self.parameter["n_classes"]).to(self.device)
        self.criterion = DiceLoss()
        self.optimizer = optim.Adam(self.model.parameters(),lr=0.0001)

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
                print(label.shape)
                print(outputs.shape)
                loss = self.criterion(outputs, label)
                print(loss)
                loss.backward()
                self.optimizer.step()

            print(loss)
            #self.test()

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


        print("accuracy:",cnt/test_num)
        print(miss_martencite/test_num)
        print(miss_ferrite/test_num)
        
if __name__ == "__main__":
    parameter = {"train_image_dir":"./data/train/image",
                 "train_label_dir":"./data/train/label",
                 "test_image_dir":"./data/test/iamge",
                 "test_label_dir":"./data/test/label",
                 "input_size":128,"epochs":100,
                 "n_classes":3,"n_chanels":3,
                 "training_rate":0.01}
    main = Main(parameter)
    main.main()
