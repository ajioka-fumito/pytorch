import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose,ToTensor 
import torch
from md.loader_gate import MyDataset
from md.model_gate import AlexNet as Model
import numpy as np

train_dataset = MyDataset("./cifar_10/train/",
                            transform = Compose([ToTensor()]))

train_ld = DataLoader(train_dataset,batch_size=100,shuffle=True)

test_dataset = MyDataset("./cifar_10/test/",transform = Compose([ToTensor()]))
test_ld = DataLoader(test_dataset)

GPU = True
device = torch.device("cuda" if GPU else "cpu")

model = Model(2).to(device).float()

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(),lr=0.0001,momentum=0.9)
losses = []

for epoch in range(5):

    for i, (image,label) in enumerate(train_ld):
        image = image.to(device).float()
        label = label.to(device).float()

        optimizer.zero_grad()
        output = model(image)

        loss = criterion(output,label)
        loss.backward()
        optimizer.step()

        if i%100==0:
            print(i,loss)

        if i%1500==0:
            losses.append(loss)
print(losses)

torch.save(model.state_dict(),"./models/gate_net/gate.pth")

def accuracy(predict,label):
    label = label.detach().numpy()
    predict = predict.cpu().detach().numpy()
    return np.argmax(label,axis=1)==np.argmax(predict,axis=1)


model = Model(2).to(device).float()
model.load_state_dict(torch.load("./models/gate_net/gate.pth"))

acc = 0
for i,(image,label) in enumerate(test_ld):
    image = image.to(device).float()
    predict = model(image)
    acc += accuracy(predict,label)
print("correct : ",acc,"all : ",i)