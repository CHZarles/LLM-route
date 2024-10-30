

import torch
import torchvision
from torch import nn
from torch.nn.modules import Sequential


# build a net work
class Classfication(nn.Module):
    def __init__(self):
        super(Classfication, self).__init__()
        self.seq = Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=5,
                padding=2,
                dilation=1,
                stride=1,
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.seq(x)


if __name__ == '__main__':
    model = Classfication()
    input = torch.ones((64, 3, 32, 32))
    output = model(input)
    print(output.shape)
