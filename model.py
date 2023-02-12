import torch.nn as nn


class Mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, input):
        return self.model(input)