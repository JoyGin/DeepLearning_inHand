import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchvision
from sklearn.model_selection import  train_test_split
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device( 'cpu')

# 获取数据
train_data = pd.read_csv('digit_recognizer/train.csv', dtype = np.float32)
test_data = pd.read_csv('digit_recognizer/test.csv', dtype = np.float32)

# print(train_data.size)
# print(test_data.size)

# 将pandas数据转化为numpy数据
train_labels = train_data.label.values
train_images = train_data.loc[:, train_data.columns != "label"].values/255
test_images  = test_data.values/255

# print('train label size:', train_labels.shape)
# print('train  image size:', train_images.shape)
# print('test image size:', test_images.shape)

# 利用sklearn把train分成train和valid
train_images, valid_images, train_labels, valid_labels =train_test_split(train_images,
                                                                         train_labels,
                                                                         test_size = 0.2,
                                             random_state = 42)


'''
train_images = Image.fromarray(train_images)   #这里ndarray_image为原来的numpy数组类型的输入

valid_images = Image.fromarray(valid_images)
# print('train size: ',train_images.shape)
# print('valid size: ',valid_images.shape)

# 更改图片的大小
resize = 
trans = []
if resize:
    trans.append(torchvision.transforms.Resize(size=resize))
transform = torchvision.transforms.Compose(trans)
train_images = transform(train_images)
# valid_images = transform(valid_images)
train_images = np.array(train_images)
# valid_images = np.array(valid_images)
print(train_images.shape)
'''
# 使用AlexNet
resize = 224
trans = []
if resize:
    trans.append(torchvision.transforms.Resize(size=resize))
    transform = torchvision.transforms.Compose(trans)
    train_imagesC = np.empty(shape = [32,224,224])
    valid_imagesC = np.empty(shape = [1,224,224])
    for i in range(train_imagesC.shape[0]):
        train_image = train_images[i]
        train_image = train_image.reshape(28,28)
        train_image = Image.fromarray(train_image)
        train_image = transform(train_image)
        train_image = np.array(train_image)
        train_imagesC[i] = train_image
    train_images = train_imagesC
    for i in range(valid_imagesC.shape[0]):
        valid_image = valid_images[i]
        valid_image = valid_image.reshape(28,28)
        valid_image = Image.fromarray(valid_image)
        valid_image = transform(valid_image)
        valid_image = np.array(valid_image)
        valid_imagesC[i] = valid_image
    valid_images = valid_imagesC


# 可视化数据
def MNshow():
    # visual
    import matplotlib.pyplot as plt
    plt.figure()
    for i in range(1, 32):
        plt.subplot(4,8,i)
        plt.imshow(train_images[i])
        plt.axis("off")
        plt.title(str(train_labels[i]))
    plt.show()

MNshow()

# 利用pytrch构建dataloader
train_images = torch.from_numpy(train_images)
train_labels = torch.from_numpy(train_labels).type(torch.LongTensor) # data type is long


valid_images = torch.from_numpy(valid_images)
valid_labels = torch.from_numpy(valid_labels).type(torch.LongTensor) # data type is long


# form dataset
train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
valid_dataset = torch.utils.data.TensorDataset(valid_images, valid_labels)


# form loader
batch_size = 64 # 2^5=64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, shuffle = True)


# 建立模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, 2),  # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

    def forward(self, img):
        features = self.conv(img)
        output = self.fc(features.view(img.shape[0], -1))
        return output


# 训练模型
model = LeNet().to(device)
# model = AlexNet().to(device)

def train():
    num_epoc = 20
    error = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epoc):
        epoc_train_loss = 0.0
        epoc_train_corr = 0.0
        epoc_valid_corr = 0.0
        print('Epoch:{}/{}'.format(epoch, num_epoc))

        for data in train_loader:
            images, labels = data
            # LeNet 训练时的长度
            if isinstance(model, LeNet):
                images = images.view(-1, 1, 28, 28)

            # AlexNet 训练时的长度
            if isinstance(model, AlexNet):
                images = images.view(-1, 1, 224, 224)
            # labels = Variable(labels)

            images = images.to(device)
            labels = labels.to(device)
            if isinstance(model, LeNet):
                outputs = model(images)
            if isinstance(model, AlexNet):
                outputs = model(images.type(torch.FloatTensor))

            optim.zero_grad()
            loss = error(outputs, labels)
            loss.backward()
            optim.step()

            epoc_train_loss += loss.data
            # epoc_train_corr += (outputs.argmax(dim=1) == labels).sum().cpu().item()
            outputs = torch.max(outputs.data, 1)[1]
            epoc_train_corr += torch.sum(outputs == labels.data)

        with torch.no_grad():
            for data in valid_loader:
                images, labels = data
                # LeNet 训练时的长度
                if isinstance(model, LeNet):
                    images = images.view(-1, 1, 28, 28)

                # AlexNet 训练时的长度
                if isinstance(model, AlexNet):
                    images = images.view(-1, 1, 224, 224)
                # labels = Variable(labels)
                images = images.to(device)
                labels = labels.to(device)

                if isinstance(model, LeNet):
                    outputs = model(images)
                if isinstance(model, AlexNet):
                    outputs = model(images.type(torch.FloatTensor))

                outputs = torch.max(outputs.data, 1)[1]

                epoc_valid_corr += torch.sum(outputs == labels.data)
                # epoc_valid_corr += (outputs.argmax(dim=1) == labels).sum().cpu().item()

        print(
            "loss is :{:.4f},Train Accuracy is:{:.4f}%,Test Accuracy is:{:.4f}".format(epoc_train_loss / len(train_dataset),
                                                                                       100 * epoc_train_corr / len(
                                                                                           train_dataset),
                                                                                       100 * epoc_valid_corr / len(
                                                                                           valid_dataset)))


def LoadToSummit():
    test_results = np.zeros((test_images.shape[0],2),dtype='int32')
    for i in range(test_images.shape[0]):
        one_image = torch.from_numpy(test_images[i]).view(1,1,28,28)
        one_image = one_image.to(device)
        one_output = model(one_image)
        test_results[i,0] = i+1
        test_results[i,1] = torch.max(one_output.data,1)[1].cpu().numpy()
    Data = {'ImageId': test_results[:, 0], 'Label': test_results[:, 1]}
    DataFrame = pd.DataFrame(Data)
    DataFrame.to_csv('./digit_recognizer/submission.csv', index=False, sep=',')


train()
# LoadToSummit()