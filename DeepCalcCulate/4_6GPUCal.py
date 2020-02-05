import torch
from torch import nn

print(torch.cuda.is_available())  # 输出 True
print(torch.cuda.device_count())  # 输出GPU个数
print(torch.cuda.current_device())  # 输出 0 GPU索引，从0开始
print(torch.cuda.get_device_name(0))  # 输出 'GeForce GTX 1050' GPU型号

# tensor的GPU计算
x = torch.tensor([1, 2, 3])
print(x)

# 使用.cuda()可以将CPU上的Tensor转换（复制）到GPU上。
# 如果有多块GPU，我们用.cuda(i)来表示
# 第 i块GPU及相应的显存（ii从0开始）且cuda(0)和cuda()等价。
# x = x.cuda(0)
# print(x)
# print(x.device)

# 在创建tensor的时候就指定设备
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# x = torch.tensor([1, 2, 3], device=device)
# or
# x = torch.tensor([1, 2, 3]).to(device)
# print(x)

# 对GPU上的数据进行计算，计算结果仍旧在GPU上
# 放在CPU上的数据不可以直接与存放在GPU上的数据进行运算，位于不同GPU上的数据也是不能直接进行计算的。
# y = x**2
# print(y)

# 模型的GPU计算
net = nn.Linear(3, 1)
# print(list(net.parameters())[0].device)

net.cuda()
# print(list(net.parameters())[0].device)

x = torch.rand(2,3).cuda()
print(net(x))
