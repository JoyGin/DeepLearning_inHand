import torch
from torch import nn
from torch.nn import init

net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))  # pytorch已进行默认初始化

# print(net)
X = torch.rand(2, 4)
Y = net(X).sum()

# 通过Module类的parameters()或者named_parameters方法来访问所有参数（以迭代器的形式返回），
# 后者除了返回参数Tensor外还会返回其名字。
# print(type(net.named_parameters()))
for name, param in net.named_parameters():
    print(name, param.size())

# for param in net.parameters():
#    print(param)

# 访问net中单层的参数。对于使用Sequential类构造的神经网络，我们可以通过方括号[]来访问网络的任一层。
# 索引0表示隐藏层为Sequential实例最先添加的层。
for name, param in net[0].named_parameters():
    print(name, param.size(), type(param))


# 如果一个Tensor是Parameter，那么它会自动被添加到模型的参数列表里，来看下面这个例子。
class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand(20, 20))
        self.weight2 = torch.rand(20, 20)

    def forward(self, x):
        pass
'''
n = MyModel()
for name, param in n.named_parameters():
    print(name)

'''
# 初始化模型参数
# 将权重参数初始化成均值为0、标准差为0.01的正态分布随机数，并依然将偏差参数清零
for name, param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01)
        print(name, param.data)
for name, param in net.named_parameters():
    if 'bias' in name:
        init.constant_(param, val=0)
        print(name, param.data)


# 自定义初始化方法
def normal_(tensor, mean=0, std=1):
    with torch.no_grad():
        return tensor.normal_(mean, std)


def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10, 10)
        tensor *= (tensor.abs() >= 5).float()

for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name, param.data)

for name, param in net.named_parameters():
    if 'bias' in name:
        param.data += 1
        print(name, param.data)


# 共享模型参数
# Module类的forward函数里多次调用同一个层。
# 此外，如果我们传入Sequential的模块是同一个Module实例的话参数也是共享的，
linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear, linear)
print(net)
for name, param in net.named_parameters():
    init.constant_(param, val=3)
    print(name, param.data)
