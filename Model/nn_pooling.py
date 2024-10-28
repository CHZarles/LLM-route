

# https://www.bilibili.com/video/av74281036?vd_source=27d3b33a76014ebb5a906ad40fa382de&spm_id_from=333.788.player.switch&p=19
# what is ceil_mode?
# True：pooling对象数量不足的时候，是否需要填充0来解决问题
# False：不填充0，直接丢弃
# pooling是一个大类，max pooling, average pooling, min pooling ....

# pooling 的作用是减少数据量的情况下保留最大特征



import torch
from torch import nn
from torch.nn import MaxPool2d

# case 1 基本使用
input = torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]], dtype=torch.float32)

print(input.shape)
input = torch.reshape(input, (-1, 1, 5, 5)) # convert to NCHW format
print(input.shape)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.pool = MaxPool2d(kernel_size=3,  ceil_mode=True)

    def forward(self, x):
        return self.pool(x)

model = Model()
res = model(input)
print(res)

import torchvision
# case 2 结合数据集
from torch.utils.data import DataLoader

# from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset , batch_size=64)

step = 0
for data in dataloader:
    imgs, targets = data 
    # writer.add_images("input", imgs, step)
    # 这里 imgs 的shape是 torch.Size([16, 3, 32, 32])
    out = model(imgs)
    # writer.add_images("output", out, step)
    step += 1

# writer.close()
