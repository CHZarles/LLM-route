import  torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from reference_code.dataloader import test_loader, targets

# get dateset
dataset = torchvision.datasets.CIFAR10(root= "./dataset" , train=False , download=True)

print("dataset's size: ", len(dataset))

# init dataloader
data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

print("last drop is True")
print("{} = {} % {}".format(len(dataset) % 64 , len(dataset), 64))

# display the batch
writer = SummaryWriter("logs")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images("Epoch: {}".format(epoch),imgs, step)
        step += 1

writer.close()

