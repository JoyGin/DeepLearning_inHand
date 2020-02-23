import math
import torch
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

features, labels = d2l.get_data_ch7()


# AdaDelta算法需要对每个自变量维护两个状态变量,，即sts t和ΔxtΔx t
def init_AdaDelta_states():
    s_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    s_b = torch.zeros(1, dtype=torch.float32)
    Dt_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    Dt_b = torch.zeros(1, dtype=torch.float32)
    # 需要与后面的zip对应
    return [[s_w, Dt_w], [s_b, Dt_b]]


def AdaDelta(params, states, hyperparams):
    lo, eps = hyperparams['lo'], 1e-5

    for p, (s, Delta)in zip(params, states):
        # print('1',p ,s ,Delta)
        s[:] = lo * s + (1 - lo) * (p.grad.data**2)
        g = torch.sqrt((Delta + eps) / (s + eps)) * p.grad.data
        p.data -= g
        Delta[:] = lo * Delta + (1 - lo) * (g**2)

# d2l.train_ch7(AdaDelta, init_AdaDelta_states(), {'lo': 0.9}, features, labels)


d2l.train_pytorch_ch7(torch.optim.Adadelta, {'rho': 0.1}, features, labels)
