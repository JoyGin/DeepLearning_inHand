import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_imgs = ImageFolder(os.path.join('../Datasets/hotdog/train'))
test_imgs = ImageFolder(os.path.join('../Datasets/hotdog/test'))

hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
# d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4);
# d2l.plt.show()

# 在训练时，我们先从图像中裁剪出随机大小和随机高宽比的一块随机区域，
# 然后将该区域缩放为高和宽均为224像素的输入。
# 测试时，我们将图像的高和宽均缩放为256像素，
# 然后从中裁剪出高和宽均为224像素的中心区域作为输入。
# 此外，我们对RGB（红、绿、蓝）三个颜色通道的数值做标准化：
# 每个数值减去该通道所有数值的平均值，
# 再除以该通道所有数值的标准差作为输出。

# 指定RGB三个通道的均值和方差来将图像通道归一化
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

test_augs = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize
    ])

pretrained_net = models.resnet18(pretrained=True)
# print(pretrained_net.fc)
#  fc是输出层函数
# 源码：self.fc = nn.Linear(512 * block.expansion, num_classes)
pretrained_net.fc = nn.Linear(512, 2)
# print(pretrained_net.fc)

output_params = list(map(id, pretrained_net.fc.parameters()))
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())

lr = 0.01
optimizer = optim.SGD([{'params': feature_params},
                       {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                       lr=lr, weight_decay=0.001)


# 微调模型
def train_fine_tuning(net, optimizer, batch_size=2, num_epochs=5):
    train_iter = DataLoader(ImageFolder(os.path.join('../Datasets/hotdog/train'), transform=train_augs),
                            batch_size, shuffle=True)
    test_iter = DataLoader(ImageFolder(os.path.join('../Datasets/hotdog/test'), transform=test_augs),
                           batch_size)
    loss = torch.nn.CrossEntropyLoss()
    d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)


train_fine_tuning(pretrained_net, optimizer)
