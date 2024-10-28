
#case1 以 ReLu 为例

import torch
from torch.nn import Module, ReLU, Sigmoid

input = torch.tensor([[1, -0.5],[-1, 3]])

input = torch.reshape(input, (-1, 1, 2,2))
print(input.shape)

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu1 = ReLU()  # Attention: 这里面有一个叫 inplace 的参数
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.relu1(input)
        output = self.sigmoid1(input)
        return output


model = Model()
output = model(input)
print(output)


#case2 
import torchvision
from torch.utils.data import DataLoader

# from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader=DataLoader(dataset, batch_size=64)

step = 0
for data in dataloader:
    imgs, targets = data 
    # writer.add_images("input", imgs, step)
    # 这里 imgs 的shape是 torch.Size([16, 3, 32, 32])
    out = model(imgs)
    # writer.add_images("output", out, step)
    step += 1

# writer.close()


