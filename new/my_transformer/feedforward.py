import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super(FeedForwardLayer, self).__init__()
        #TODO two lines!
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        #TODO one line!
        return self.ff2(self.ff1(x))

class DropoutLayer(nn.Module):
    def __init__(self, p: float) -> None:
        super(DropoutLayer, self).__init__()
        self.dropout = nn.Dropout(p)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x)

class ActivationLayer(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        #TODO one line!
        return F.relu(x)