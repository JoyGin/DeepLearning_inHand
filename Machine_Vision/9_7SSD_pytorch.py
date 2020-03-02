import sys
sys.path.insert(0, '..')
import d2l
# from d2l.ssd_utils import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import json
import time
from tqdm import tqdm
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cls_predictor(input_channels, num_anchors, num_classes):
    return nn.Conv2d(in_channels=input_channels, out_channels=num_anchors * (num_classes + 1), kernel_size=3,
                     padding=1)


def bbox_predictor(input_channels, num_anchors):
    return nn.Conv2d(in_channels=input_channels, out_channels=num_anchors * 4, kernel_size=3, padding=1)


def forward(x, block):
    return block(x.to(device))


Y1 = forward(torch.zeros((2, 8, 20, 20)).to(device), cls_predictor(8, 5, 10).cuda())
Y2 = forward(torch.zeros((2, 16, 10, 10)).to(device), cls_predictor(16, 3, 10).cuda())
print(Y1.shape, Y2.shape)


def flatten_pred(pred):
    return pred.permute(0, 2, 3, 1).reshape(pred.size(0),-1)

def concat_preds(preds):
    return torch.cat(tuple([flatten_pred(p) for p in preds]), dim=1)