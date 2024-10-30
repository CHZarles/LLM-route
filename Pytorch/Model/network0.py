import torch
from torch import nn
from torch.nn.modules import Sequential

# CRFAR10 dateset   



# a classficition model
class Classfication(nn.Module):
    def __init__(self):
        super(Classfication,self).__init__()
        '''
        Conv2d
        - input: (N , C_in , H_in , W_in)
        - output: (N, C_out, H_out , W_out)

        H_out = [ H_in + 2*Padding[0] - dilation[0] * (kernel_size[0] - 1) - 1 ] / stride[0] + 1
        W_out = [ W_in + 2*Padding[1] - dilation[0] * (kernel_size[1] - 1) - 1 ] / stride[1] + 1

        '''
        # Layer0 
        # Conv2d,  kernel (5,5) ,  shape:  (3 , 32,  32 ) -> (32, 32, 32)
        # 32 = (32 + 2*Padding[0] - dilation[0] * (5 - 1) - 1)/stride[0] + 1
        # => Padding[0] = 2, dilation[0] = 1 , stride[0] = 1
        self.layer1 = nn.Conv2d(in_channels=3,out_channels=32, kernel_size=5,padding=2,dilation=1,stride=1)
        # Layer1
        # Max-pooling , kernel (2,2)  shape (32, 32, 32) -> (32, 16, 16)
        self.layer2 = nn.MaxPool2d(kernel_size=2)
        # layer2
        # Conv2d, kernel (5, 5) , shape  (32, 16, 16) -> (32, 16, 16)
        self.layer3 =  nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2) # dilation and stride default 1
        # layer3
        # Max-pooling , kernel (2,2) shape (32, 16, 16) -> (32, 8 , 8)
        self.layer4 = nn.MaxPool2d(kernel_size=2) 
        # layer4
        # Conv2d , kernel (5,5) ,  shape (32, 8 , 8)  -> (64 , 8 , 8)
        self.layer5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2 )
        # layer5 
        # Max-pooling , kernel (2,2) , shape (64, 8, 8) -> (64, 4, 4)
        self.layer6 = nn.MaxPool2d(kernel_size=2)
        # layer6 , 7, 8
        # Flatten + linear
        self.layer7 = nn.Flatten() # (64,4,4) -> (1024,)
        self.layer8 = nn.Linear(1024, 64) # (1024,) ->  (64,)
        self.layer9 = nn.Linear(64, 10) # (64,) ->  (10,)

 

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        return x

# 改进版
class Classfication2(nn.Module):
    def __init__(self):
        super(Classfication2,self).__init__()
        self.seq = Sequential(
        nn.Conv2d(in_channels=3,out_channels=32, kernel_size=5,padding=2,dilation=1,stride=1),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2) ,
        nn.MaxPool2d(kernel_size=2) ,
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2 ),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten() ,
        nn.Linear(1024, 64) ,
        nn.Linear(64, 10) ,
        ) 

    def forward(self, x):
        return self.seq(x)



model = Classfication()
print(model)
input = torch.ones((64,3,32,32))
output = model(input)
print(output.shape)

        
model = Classfication2()
output = model(input)
print(output.shape)


'''
Display in TensorBoard

....
writer.add_garph(model,input)
....

'''
