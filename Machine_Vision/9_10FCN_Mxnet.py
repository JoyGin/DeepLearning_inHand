import d2lzh as d2l
from mxnet import gluon, image, init, nd
from mxnet.gluon import data as gdata, loss as gloss, model_zoo, nn
import mxnet as mx
import numpy as np
import sys


X = nd.arange(1, 17).reshape((1, 1, 4, 4))
K = nd.arange(1, 10).reshape((1, 1, 3, 3))
conv = nn.Conv2D(channels=1, kernel_size=3)
conv.initialize(init.Constant(K))
# print(X,conv(X),K)

W, k = nd.zeros((4, 16)), nd.zeros(11)
k[:3], k[4:7], k[8:] = K[0, 0, 0, :], K[0, 0, 1, :], K[0, 0, 2, :]
W[0, 0:11], W[1, 1:12], W[2, 4:15], W[3, 5:16] = k, k, k, k
# print(nd.dot(W, X.reshape(16)).reshape((1, 1, 2, 2)), W)

conv = nn.Conv2D(10, kernel_size=4, padding=1, strides=2)
conv.initialize()

X = nd.random.uniform(shape=(1, 3, 64, 64))
Y = conv(X)
# print(Y.shape)


conv_trans = nn.Conv2DTranspose(channels=3, kernel_size=4, padding=1, strides=2)
conv_trans.initialize()
# print(conv_trans(Y).shape)

pretrained_net = model_zoo.vision.resnet18_v2(pretrained=True)
# print(pretrained_net.features[-4:], pretrained_net.output)

net = nn.HybridSequential()
for layer in pretrained_net.features[:-2]:
    net.add(layer)

X = nd.random.uniform(shape=(1, 3, 320, 480))
# print(net(X).shape)


num_classes = 21
net.add(nn.Conv2D(num_classes, kernel_size=1),
        nn.Conv2DTranspose(num_classes, kernel_size=64, padding=16,
                           strides=32))


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return nd.array(weight)