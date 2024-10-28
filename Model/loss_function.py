# what is loss function?
"""
Loss function

1. 计算预测值和真实值之间的差异
2. 为我们更新输出提供一定的依据（反向传播

"""


import torch
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

# function 1
loss = L1Loss(reduction="sum")
result = loss(inputs, targets)

print(result)


# function 2
loss_mse = MSELoss()
result_mse = loss_mse(inputs, targets)
print(result_mse)


# function 3
# CrossEntropyLoss,针对多分类问题，计算交叉熵损失
x = torch.tensor([0.1, 0.2, 0.3])  # 三个类别的预测值
y = torch.tensor([1])  # 这里计算class 1
x = torch.reshape(x, (1, 3))  # reshape to (N, C, ... ), C 是类别数
loss_cross = CrossEntropyLoss()
print(loss_cross(x, y))  # 输出tensor(1.1019)
