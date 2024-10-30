
import torch
from torch import nn


class Person(nn.Module):
    def __init__(self, name, age):
        super().__init__()
        self.name = name
        self.age = age
    
    def forward(self, x):
        self.age += x
        return self.age


man = Person('John', 20)
x = torch.tensor(5)
out = man(x)
print(type(out), out)
