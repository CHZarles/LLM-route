import torch
import torchvision
from torch import nn
from torch.nn.modules import Sequential

# CRFAR10 dateset


# 改进版
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
for data in dataloader:
    img, targets = data
    output = model(img)
    result_loss = loss(output, targets)
    print(result_loss)  # 结果包含grad
    result_loss.backward()  # 反向传播
