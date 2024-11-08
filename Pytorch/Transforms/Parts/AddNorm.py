import torch
from torch import nn

# https://www.bilibili.com/video/BV11V4y177vr?spm_id_from=333.788.videopod.sections&vd_source=27d3b33a76014ebb5a906ad40fa382de

# Layer Normalization

# 实现标准化的公式
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # self.a_2 和 self.b_2 是可学习的参数
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps


    # 确保传入的x的最后的维度是feature的大小
    def forward(self, x):
        """
        在这个上下文中，`-1` 表示最后一个维度。Python 中的负索引允许你从数组或张量的末尾开始计数。
        `keepdim` 是一个布尔参数，用于在计算张量的均值（或其他归约操作）时，决定是否保留被归约的维度。
        - `keepdim=True`：保留被归约的维度，维度的大小变为1。
        - `keepdim=False`（默认）：不保留被归约的维度，结果张量的维度会减少。
        """
        print(" ================== In LayerNorm.forward ================== ")
        mean = x.mean(-1, keepdim=True)
        print("mean.shape:\n", mean.shape)
        std = x.std(-1, keepdim=True)
        print("std.shape:\n", std.shape)
        print("x.shape:\n", x.shape)
        # print("self.a_2.shape:\n", self.a_2.shape)
        # print("self.b_2.shape:\n", self.b_2.shape)
        # print("self.eps:\n", self.eps)
        # print("x: " , x)
        # print("mean:\n", mean)
        # print("x - mean:\n", x - mean)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2





if __name__ == "__main__":
    # 
    feature = torch.randn(10 , 4)  
    # demostrate the LayerNorm
    layerNorm = LayerNorm(4)
    # Apply LayerNorm to the feature tensor
    normalized_feature = layerNorm(feature)
    # Print the original and normalized features
    # print("Original Feature:\n", feature)
    # print("Normalized Feature:\n", normalized_feature)
