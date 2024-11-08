import math

import torch
from torch import nn
from torch.nn import functional as F


# implement self attention mechanism
# query，key，value 同源，才叫 self attention
# 为了方便后续实现 mask attention，我们将 mask 作为参数传入
def self_attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)
    # matmul and scale
    print(" ============== self_attention ==============")
    print("q.size():", q.size())
    print("k.transpose(-2,-1).size():", k.transpose(-2,-1).size())
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    print("scores.size()", scores.size())
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None: # dropout is not necessary,训练时可以用，测试时不用
        scores = dropout(scores)
    return torch.matmul(scores, v), scores


if __name__ == "__main__":
    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 3, 4)
    v = torch.randn(2, 3, 4)
    mask=torch.tensor(
        [
            [[1, 0, 1], [1, 1, 0], [1, 1, 0]],
            [[1, 0, 1], [1, 1, 0], [1, 1, 0]]
        ])
    out, scores = self_attention(q, k, v, mask)
    print(out)
    print(scores)
    # print(out.size(), scores.size())
    # print(out.size(-1), scores.size(-1))
