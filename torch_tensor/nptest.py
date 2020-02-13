import numpy as np
from PIL import Image
import torchvision
import torch

X = torch.tensor([
    [5,2,3,4],
    [5,6,9,8]
])
print(X.argmax(dim=1))
'''
X = np.random.rand(2,28,28)
Y = np.empty(shape=[2,32,32])
print(X[0])
trans = []
resize = (32,32)
trans.append(torchvision.transforms.Resize(size=resize))
transform = torchvision.transforms.Compose(trans)
Y = np.empty(shape=[2,32,32])
for i in range(X.shape[0]):
    train_image = Image.fromarray(X[i])
    train_image = transform(train_image)
    Y[i] = np.array(train_image)
print('Y[0]:')
print(Y[0])
'''