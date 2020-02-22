import d2lzh_pytorch as d2l
import torch

eta = 0.4 # 学习率


def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)


# d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
# eta = 0.6
# d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))


def momentum_2d(x1, x2, v1, v2):
    v1 = gamma * v1 + eta * 0.2 * x1
    v2 = gamma * v2 + eta * 4 * x2
    return x1 - v1, x2 - v2, v1, v2

# eta, gamma = 0.6, 0.5
# d2l.set_figsize((10, 10))
# d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))


features, labels = d2l.get_data_ch7()


def init_momentum_states():
    v_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    v_b = torch.zeros(1, dtype=torch.float32)
    return (v_w, v_b)


def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v.data = hyperparams['momentum'] * v.data + hyperparams['lr'] * p.grad.data
        p.data -= v.data

# d2l.train_ch7(sgd_momentum, init_momentum_states(),{'lr': 0.004, 'momentum': 0.9}, features, labels)


d2l.train_pytorch_ch7(torch.optim.SGD, {'lr': 0.004, 'momentum': 0.9},
                    features, labels)
