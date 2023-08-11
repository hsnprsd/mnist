import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, n, num_classes) -> None:
        super().__init__()

        conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv = nn.Sequential(conv1, pool1, conv2, pool2, nn.ELU())

        ffwd_input = (n // 4) ** 2 * 64
        self.ffwd = nn.Sequential(
            nn.Linear(ffwd_input, 128),
            nn.ELU(),
            nn.Linear(128, num_classes),
            nn.Softmax(),
        )

    def forward(self, x):
        B = x.shape[0]
        out = self.conv(x).view(B, -1)
        out = self.ffwd(out)
        return out
