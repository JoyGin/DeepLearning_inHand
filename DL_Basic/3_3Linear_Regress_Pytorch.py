import torch
import numpy as np
import torch.utils.data as Data
from torch.nn import init
from collections import OrderedDict
import torch.nn as nn
import torch.optim as optim

# 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)


# 读取数据
batch_size = 10
dataset = Data.TensorDataset(features, labels) # 将训练数据的特征和标签组合
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True) # 随机读取小批量
# print(dataset.tensors)
'''
for i in dataset:
    print(i)

for X, y in data_iter:
    print(X, y)
    break
'''

'''
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)
    def forward(self, x):
        y = self.linear(x)
        return y
    
net = LinearNet(num_inputs)
print(net) # 使用print可以打印出网络的结构

# 写法一
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
    )

# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......

# 写法三
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
          ('linear', nn.Linear(num_inputs, 1))
          # ......
        ]))

print(net)
print(net[0])
'''
# 定义模型
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y


net = LinearNet(num_inputs)
print(net)
# print(net[0])



# 初始化模型参数
# print('weight:',net[0].weight)
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)


# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
optimizer = optim.SGD(net.parameters(), lr=0.03)
'''
for i in net.parameters():
    print(i)
'''
# 调整学习率
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1 # 学习率为之前的0.1倍

# 训练模型
num_epochs = 4
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        # print(l)
        # print(optimizer)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
    print('epoch %d, loss: %f' % (epoch, l.item()))


dense = net.linear
print(true_w,'\n', dense.weight)
print(true_b, '\n',dense.bias)