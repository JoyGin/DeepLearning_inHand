import torch
import d2lzh_pytorch as d2l

features, labels = d2l.get_data_ch7()


def init_Adam_states():
    v_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    v_b = torch.zeros(1, dtype=torch.float32)
    s_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    s_b = torch.zeros(1, dtype=torch.float32)
    # 需要与后面的zip对应
    return [[s_w, v_w], [s_b, v_b]]


def Adam(params, states, hyperparams):
    pha1, pha2, eps = hyperparams['pha1'],hyperparams['pha2'], 1e-5
    t = hyperparams['t']
    lr = hyperparams['lr']
    for p, (s, v)in zip(params, states):
       v = pha1 * v + (1 - pha1) * p.grad.data
       s = pha2 * s + (1 - pha2) * (p.grad.data**2)
       v_t = v / (1 - (pha1**t))
       s_t = s / (1 - (pha2**t))
       g = (lr * v_t) / torch.sqrt(s_t + eps)
       p.data -= g
    hyperparams['t'] += 1
# d2l.train_ch7(Adam, init_Adam_states(), {'pha1': 0.9, 'pha2':0.999, 't':1, 'lr': 0.01}, features, labels)


d2l.train_pytorch_ch7(torch.optim.Adam, {'lr': 0.01}, features, labels)
