

# https://yey.world/2020/12/16/Pytorch-13/
# network layer

# 25 * 25 的层变 3 * 3 的层

import torch
import torchvision
from torch.nn import Linear, Module
from torch.utils.data import DataLoader

# from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset , batch_size=64,drop_last=True )

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = Linear(196608,10)

    def forward(self, input):
        output = self.linear1(input)
        return output

model = Model()
for data in dataloader:
    imgs , targes = data
    print(imgs.shape)
    # output = torch.reshape(imgs,(1,1,1,-1)) # why ?
    output = torch.flatten(imgs) # 效果同上注释
    print(output.shape)
    output = model(output)
    print(output)

