import torch

input = torch.tensor([[1,2,0,3,1],[0,1,2,3,1],[1,2,1,0,0],[5,2,3,1,1],[2,1,0,1,1]])

kernel = torch.tensor([[1,2,1],[0,1,0],[2,1,0]])
print(input)
print(kernel)



import torch.nn.functional as F

# conv2d receive 3D input and kernel, so reshape them 
input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))


print(input.shape)
print(kernel.shape)

# padding 就是在原图像周围填充0的层数
out = F.conv2d(input, kernel, stride=1, padding=0)
print(out)

out = F.conv2d(input, kernel, stride=2, padding=0)
print(out)

# Q: 卷积如果越界了，会怎么样？越界的部分好像会被忽略
out = F.conv2d(input, kernel, stride=3, padding=0)
print(out)

