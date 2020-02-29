import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mxnet.gluon import utils as gutils  # pip install mxnet==0.11.0
from mxnet import image

data_dir = '../Data/pikachu'
os.makedirs(data_dir, exist_ok=True)


def _download_pikachu(data_dir):
    root_url = ('https://apache-mxnet.s3-accelerate.amazonaws.com/'
                'gluon/dataset/pikachu/')
    dataset = {'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
               'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
               'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'}
    for k, v in dataset.items():
        gutils.download(root_url + k, os.path.join(data_dir, k), sha1_hash=v)

if not os.path.exists(os.path.join(data_dir, "train.rec")):
    print("下载原始数据集到%s..." % data_dir)
    _download_pikachu(data_dir)


def load_data_pikachu(batch_size, edge_size=256):  # edge_size：输出图像的宽和高
    train_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, 'train.rec'),
        path_imgidx=os.path.join(data_dir, 'train.idx'),
        batch_size=batch_size,
        data_shape=(3, edge_size, edge_size),  # 输出图像的形状
        # shuffle=False,  # 以随机顺序读取数据集
        # rand_crop=1,  # 随机裁剪的概率为1
        min_object_covered=0.95, max_attempts=200)

    val_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, 'val.rec'),
        batch_size=batch_size,
        data_shape=(3, edge_size, edge_size),
        shuffle=False)
    return train_iter, val_iter


batch_size, edge_size = 1, 256
train_iter, val_iter = load_data_pikachu(batch_size, edge_size)
batch = train_iter.next()
# batch.data[0][0].shape, batch.label[0][0].shape


def process(data_iter, save_dir):
    """batch size == 1"""
    data_iter.reset() # 从头开始
    all_label = dict()
    id = 1
    os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
    for sample in tqdm(data_iter):
        x = sample.data[0][0].asnumpy().transpose((1,2,0))
        plt.imsave(os.path.join(save_dir, 'images', str(id) + '.png'), x / 255.0)

        y = sample.label[0][0][0].asnumpy()

        label = {}
        label["class"] = int(y[0])
        label["loc"] = y[1:].tolist()

        all_label[str(id) + '.png'] = label.copy()

        id += 1

    with open(os.path.join(save_dir, 'label.json'), 'w') as f:
        json.dump(all_label, f, indent=True)

process(data_iter = train_iter, save_dir = os.path.join(data_dir, "train"))
process(data_iter = val_iter, save_dir = os.path.join(data_dir, "val"))
