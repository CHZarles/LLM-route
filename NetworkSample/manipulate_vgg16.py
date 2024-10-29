import torch
import torchvision
from torch import nn

# download image_net
# RuntimeError: The archive ILSVRC2012_devkit_t12.tar.gz is not present in the root directory or is corrupte
# d. You need to download it externally and place it in data.
# train_data = torchvision.datasets.ImageNet(root='data', train=True, download=True)



'''知识点1： 模型的下载'''
# import VGG16 model from torch
vgg16_trained = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1, progress=True)
vgg16_raw = torchvision.models.vgg16(progress=True)
assert not torch.equal(vgg16_trained.classifier[0].weight, vgg16_raw.classifier[0].weight), "The weights are equal"



'''知识点2： 模型的修改'''
# image net 有1000个类别, 现在想适配到自己的只有10个类别的数据集CIFAR10
# 增加一个的全连接层
vgg16_trained.classifier.add_module("add_linear", nn.Linear(1000,10))
vgg16_trained.classifier[7] = nn.Linear(1000,10)
vgg16_trained.classifier._modules.popitem()


'''知识点3： 模型的保存'''
torch.save(vgg16_raw, "vgg16_method1.pth")  #  save model structure and model's parameter
torch.save(vgg16_trained.state_dict(), "vgg16_method2.pth") # only save the model 's parameter

'''知识点4： 模型的加载'''
model_raw = torch.load("vgg16_method1.pth")
model_trained_parameter = torch.load("vgg16_method2.pth") 
model_raw.load_state_dict(model_trained_parameter)

assert torch.equal(model_raw.classifier[0].weight, vgg16_trained.classifier[0].weight), "The weights are not equal"
