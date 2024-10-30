from PIL import Image
from torchvision import  transforms
from torch.utils.tensorboard import  SummaryWriter



writer = SummaryWriter("logs")

img=Image.open ("./pets.jpg")
# 常见transform
# 1. ToTensor
to_tensor = transforms.ToTensor()
img_tensor = to_tensor(img)
writer.add_image("ToTensor" ,img_tensor)

# 2. Normalize
# 均值是什么，标准差是什么
# input[channel] = (input[channel] - mean[channel]) / std[channel]
writer.add_image("Normalize" ,img_tensor)
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 1)

trans_norm = transforms.Normalize([0.5,0.5,5],[0.5,0.5,5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 2)


# 3. resize
print("original size:" , img.size)
img_resizer = transforms.Resize((256, 256))
img_resize = img_resizer(img)
img_tensor = to_tensor(img_resize)
writer.add_image("Resize", img_tensor, 0)

# 4. compose
# 这个类型接受的参数是 transform 操作的列表 ops
# 但是 ops[i] 的输出一定要匹配 ops[i+1]的输入
img_resizer = transforms.Resize(512)
ops = [img_resizer, to_tensor]
trans_compose = transforms.Compose(ops)
img_compose = trans_compose(img)
writer.add_image("Resize", img_compose, 1)

# 5. RandomCrop
trans_random = transforms.RandomCrop((128,512))
ops = [trans_random, to_tensor]
trans_compose = transforms.Compose(ops)
for i in range(10):
    writer.add_image("Random Crop" , trans_compose(img) , i)

writer.close()


# Summary
# 1. 关注输入和输出 2. 查官方文档