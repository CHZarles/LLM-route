import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from custom_model import Classfication

train_data = torchvision.datasets.CIFAR10(
    "./data", train=True, download=True, transform=torchvision.transforms.ToTensor()
)
test_data = torchvision.datasets.CIFAR10(
    "./data", train=False, download=True, transform=torchvision.transforms.ToTensor()
)

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

# 定义训练的设备
# device = torch.device("cpu")
# device = torch.device("cuda")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# create mode
model = Classfication()
model = model.to(device)

# define loss function
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# init a writer
writer = SummaryWriter("logs")

# Optimizer
learning_rate = 0.01  # 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# config train parameters
total_train_step = 0
total_test_step = 0
epochs = 20


# 课后作用：将训练过程写在tensorboarn
for i in range(epochs):
    model.train()  # set model to train mode, 这个针对某些层会生效，反正调用也没问题
    print(" ----- epoch {} : begin training -----".format(i))
    # load data
    for data in train_data_loader:
        imgs, targets = data

        # use cuda
        imgs = imgs.to(device)
        targets = targets.to(device)

        ouputs = model(imgs)
        loss_val = loss_fn(ouputs, targets)
        # update parameters, back propagation
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print(" ----- epoch {} , loss : {} -----".format(i, loss_val.item()))
            writer.add_scalar("train_loss", loss_val.item(), total_train_step)

    # test performance
    model.eval()  # set model to eval mode
    total_test_loss = 0
    with torch.no_grad():
        for date in test_data_loader:
            imgs, targets = data
            # use cuda
            imgs = imgs.to(device)
            targets = targets.to(device)
            ouputs = model(imgs)
            loss_val = loss_fn(ouputs, targets)
            total_test_loss += loss_val
    total_test_step += 1
    print(" ----- epoch {} , test loss : {} -----".format(i, total_test_loss))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
