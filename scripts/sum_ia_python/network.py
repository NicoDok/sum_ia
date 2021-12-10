import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.l1 = nn.Linear(1568, 392)
        self.relu1 = nn.ReLU() # casse la linearite
        self.l2 = nn.Linear(392, 98)
        self.relu2 = nn.ReLU() # casse la linearite
        self.l3 = nn.Linear(98, 19)
        
    def forward(self, x):
        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)
        x = self.relu2(x)
        x = self.l3(x)
        return F.softmax(x, dim=1)