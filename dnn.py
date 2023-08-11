import torch.nn as nn
import torch.nn.functional as F


class MNIST(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.nn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        out = F.softmax(self.nn(x))
        return out
