import PIL.Image
import torch
import torchvision
#t 应该每个数据集的用法都不一样，还是要看文档

# 引入数据集
train_set = torchvision.datasets.CIFAR10(root = "./dataset", train=True , download=True)
test_set = torchvision.datasets.CIFAR10(root= "./dataset" , train=False , download=True)

PIL_img , label_idx_0 = test_set[0]
assert isinstance(PIL_img, PIL.Image.Image)
# 用 Transform （不同的数据集用法不一样，看文档行事
from torch.utils.tensorboard import SummaryWriter
data_set_transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()]
)

test_set_1 = torchvision.datasets.CIFAR10(root= "./dataset" , transform=data_set_transform, train=False , download=True)
tensor_img , label_idx_1 = test_set_1[0]
assert isinstance(tensor_img, torch.Tensor)

assert label_idx_1 == label_idx_0