import torch
from torch import nn,optim
from md.loader import Mydataset
from md.model import Linear_model
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main(param):
    # gpu or cpu
    GPU = True
    device = torch.device("cuda" if GPU == True else "cpu")
    train_data,test_data = train_test_split(Mydataset(parameter["path"]))
    
    # train and validation 
    train_loader = DataLoader(train_data,batch_size=parameter["batch_size"],shuffle=False)
    test_loader  =  DataLoader(test_data,batch_size=5,shuffle=False)
    # model
    model = Linear_model(parameter["input_size"],parameter["output_size"]).float().to(device)
    # loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(),lr=parameter["learning_rate"])

    # statistic object
    loss_score = []
    train_accuracy = []
    test_accuracy = {}
    
    for epoch in range(parameter["num_epochs"]):
        for i,(dx,dt) in enumerate(train_loader):
            optimizer.zero_grad()
            dx = dx.float().to(device)
            dt = dt.float().to(device)
            dy = model(dx)
            loss = criterion(dy,dt)
            loss.backward()
            optimizer.step()

        if epoch%5==1:
            print("epoc {:02d}/{:02d} : Loss:{:.3f}".format(epoch+1,
                                                            parameter["num_epochs"],
                                                            loss.item()))
            loss_score.append(loss.item())
            test_loss = []
            for i,(test_x,test_t) in enumerate(test_loader):
                test_x = test_x.float().to(device)
                test_t = test_t.float().to(device)
                test_y = model(test_x)
                test_loss.append(criterion(test_y,test_t))
            # need to revice
            test_accuracy[epoch+1] = 1-sum(test_loss)/5

    plt.plot(test_accuracy.keys(),test_accuracy.values())
    plt.show()

    torch.save(model.state_dict(),"./result/model.pkl")

if __name__ == "__main__":
    
    parameter = {"path":"./data/koos234.csv",
                 "input_size":22,
                 "output_size":1,
                 "num_epochs":100,
                 "batch_size":10,
                 "learning_rate":0.001}
    
    main(parameter)
    
    