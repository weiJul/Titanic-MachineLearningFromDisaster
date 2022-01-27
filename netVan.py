import torch
import torch.nn as nn
import torch.nn.functional as F

class NetVan(nn.Module):
    def __init__(self, inputLayer, outputLayer):
        super().__init__()
        self.linear1 = nn.Linear(inputLayer,800)
        self.linear2 = nn.Linear(800,12)
        self.linear3 = nn.Linear(12,outputLayer)
        self.bn2 = nn.BatchNorm1d(12)
        self.bn1 = nn.BatchNorm1d(800)
        self.dr = nn.Dropout(p=0.3)

    def forward(self,x):
        x = F.relu6(self.bn1(self.linear1(x)))
        x = self.dr(x)
        x = F.relu6(self.bn2(self.linear2(x)))
        x = self.dr(x)
        x = self.linear3(x)

        return torch.sigmoid(x)