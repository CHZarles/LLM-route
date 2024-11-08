import torch
from torch import nn

from AddNorm import LayerNorm


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout) # not necessary 

    def forward(self, x, sublayer):
        # 这里 sublayer(x) 对应图里面的Z, 那么sublayer 就对应图里的 self-attention 或者 feed forward
        # 保证 x最后的维度符合初始化的size
        print(" ================== In SublayerConnection.forward ================== ")
        print("x.shape:\n", x.shape)
        print("sublayer(x).shape:\n", sublayer(x).shape)
        return self.dropout(self.norm(x + sublayer(x)))





import unittest


class TestSublayerConnection(unittest.TestCase):
    def setUp(self):
        self.size = 512
        self.dropout = 0.1
        self.sublayer_connection = SublayerConnection(self.size, self.dropout)
        self.input_tensor = torch.randn(10, self.size)
        self.sublayer = nn.Linear(self.size, self.size)

    def test_forward(self):
        print("input tensor.shape:\n", self.input_tensor.shape)
        output = self.sublayer_connection(self.input_tensor, self.sublayer)
        print("output tensor.shape:\n", output.shape)
        self.assertEqual(output.shape, self.input_tensor.shape)


if __name__ == "__main__":
    # test the SublayerConnection
    unittest.main()
