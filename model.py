import torch
import torch.nn as nn

class ContrastiveModel(nn.Module):
    def __init__(self):
        super(ContrastiveModel, self).__init__()

        self.start = nn.Linear(100, 20)

    def forward(self, x):
        print(x)
        return x