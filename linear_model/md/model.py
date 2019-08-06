from torch import nn

class Linear_model(nn.Module):
    def __init__(self,input_size,output_size):
        super(Linear_model, self).__init__()
        self.Linear = nn.Linear(input_size,output_size)
        
    def forward(self,x):
        out = self.Linear(x)
        return out
