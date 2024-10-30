import torch
import torchvision
from torch import nn
from torch.nn import Conv2d

# from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        return self.conv1(x)




mymodel = Model()
# display model structure
# print(mymodel)
step = 0
# writer = SummaryWriter("logs")
for data in dataloader:
    img, targets = data
    # torch.Size([64, 3, 32, 32])
    print(img.shape)
    out = mymodel(img)
    # torch.Size([64, 6, 30, 30])
    print(out.shape)

    # writer.add_images("input", img, step)
    # writer.add_images("out", out, step)

    step += 1

# writer.close()


'''
具体参数 https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

遗留问题：

1.上面设置kernel是3，kernel的具体内容是什么？？

2.尚未清楚的参数用法
stride
dilation

https://github.com/vdumoulin/conv_arithmetic


'''
