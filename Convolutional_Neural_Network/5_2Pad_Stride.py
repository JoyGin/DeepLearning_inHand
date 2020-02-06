import torch
from torch import nn


# 定义一个函数来计算卷积层。它对输入和输出做相应的升维和降维
def comp_conv2d(conv2d, X):
    # (1, 1)代表批量大小和通道数（“多输入通道和多输出通道”一节将介绍）均为1
    X = X.view((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:])  # 排除不关心的前两维：批量和通道

# 注意这里是两侧分别填充1行或列，所以在两侧一共填充2行或列
# conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

X = torch.tensor([
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]
                ])
w = torch.tensor([
    [0,1],
    [2,3]
                ])
conv2d = nn.Conv2d(in_channels = 1,out_channels = 1, kernel_size=(2, 2), stride=(3, 2),padding = (1,1))
print(comp_conv2d(conv2d, X.float()))
