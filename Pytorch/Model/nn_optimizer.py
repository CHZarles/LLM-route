
import torch
import torchvision
from torch import nn
from torch.nn.modules import Sequential

'''
Optimizers , 不同的优化器有不同的参数
基本使用
1. 设置model的parameters, 学习率
2. 在训练中和loss结合使用，清0梯度，反向传播，更新参数

'''

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


model = Classfication()
dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(20): # 20个epoch,学习次数
    running_loss = 0.0
    for data in dataloader:
        img, targets = data
        output = model(img)
        result_loss = loss(output, targets)
        # print(result_loss)  # 结果包含grad  
        # !!! 
        optim.zero_grad()  # 清0梯度
        result_loss.backward()  # 反向传播
        optim.step()  # 更新参数

        running_loss += result_loss
    print(f"epoch {epoch} loss: {running_loss}")
