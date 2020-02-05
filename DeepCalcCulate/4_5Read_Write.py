import torch
from torch import nn


# 存储在disk
x = torch.ones(3)
torch.save(x, 'x.pt')
x2 = torch.load('x.pt')

# 存储一个列表并且返回
y = torch.zeros(4)
torch.save([x, y], 'xy.pt')
xy_list = torch.load('xy.pt')
xy_list

# 存取一个字符串映射到tensor的字典
torch.save({'x': x, 'y': y}, 'xy_dict.pt')
xy = torch.load('xy_dict.pt')


# 读写模型
# Module的可学习参数(即权重和偏差)，
# 模块模型包含在参数中(通过model.parameters()访问)。
# state_dict是一个从参数名称隐射到参数Tesnor的字典对象。
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

# 注意，只有具有可学习参数的层(卷积层、线性层等)才有state_dict中的条目。
# 优化器(optim)也有一个state_dict，其中包含关于优化器状态以及所使用的超参数的信息。
net = MLP()
net.state_dict()

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer.state_dict()

# 保存和加载模型
# PyTorch中保存和加载训练模型有两种常见的方法:
# 1.仅保存和加载模型参数(state_dict)；
# 2.保存和加载整个模型。

# 1
# 保存
# torch.save(model.state_dict(), PATH) # 推荐的文件后缀名是pt或pth
# 加载
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))

# 2
# 保存
# torch.save(model, PATH)
# 加载
# model = torch.load(PATH)
X = torch.randn(2, 3)
Y = net(X)

PATH = "./net.pt"
torch.save(net.state_dict(), PATH)

net2 = MLP()
net2.load_state_dict(torch.load(PATH))
Y2 = net2(X)
Y2 == Y
