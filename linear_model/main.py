import torch
from torch import nn,optim
from md.loader import Mydataset
from md.model import Linear_model
from torch.utils.data import DataLoader


def main(param):
    # gpu or cpu
    GPU = True
    device = torch.device("cuda" if GPU == True else "cpu")
    dataset = Mydataset(parameter["path"])
    
    # model
    model = Linear_model(parameter["input_size"],parameter["output_size"]).float().to(device)
    # loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=parameter["learning_rate"])
    
    for epoch in range(parameter["num_epochs"]):

        loader = DataLoader(dataset,batch_size=parameter["batch_size"],shuffle=False)

        for i,(dx,dt) in enumerate(loader):
            
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
        torch.save(model.state_dict(),"./result/model.pkl")

if __name__ == "__main__":
    
    parameter = {"path":"./data/koos234.csv",
                 "input_size":22,
                 "output_size":1,
                 "num_epochs":100,
                 "batch_size":1,
                 "learning_rate":0.001}
    
    main(parameter)
    
    