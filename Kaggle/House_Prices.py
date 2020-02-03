import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l


train_data = pd.read_csv('house_prices/train.csv')
test_data = pd.read_csv('house_prices/test.csv')

'''
第一个特征是Id，它能帮助模型记住每个训练样本，
但难以推广到测试样本，所以我们不使用它来训练。
我们将所有的训练数据和测试数据的79个特征按样本连结。
'''

all_features = pd.concat(train_data.iloc[:, 1:-1], test_data.iloc[:, 1:])