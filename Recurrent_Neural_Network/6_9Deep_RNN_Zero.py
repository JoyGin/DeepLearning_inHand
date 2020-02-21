import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取数据集
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

num_inputs, num_hiddens, num_outputs, Deep_Layer = vocab_size, 256, vocab_size, 3
print('will use', device)


def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device, requires_grad=True))


    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, requires_grad=True))

    W_txh = _one((num_hiddens, num_hiddens))
    W_thh = _one((num_hiddens, num_hiddens))
    b_th = torch.nn.Parameter(torch.zeros(num_hiddens, device=device, requires_grad=True))
    return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q, W_txh, W_thh, b_th])


def Deep_Layer_Param():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)

    W_txh = _one((num_hiddens, num_hiddens))
    W_thh = _one((num_hiddens, num_hiddens))
    b_th = torch.nn.Parameter(torch.zeros(num_hiddens, device=device, requires_grad=True))

    return nn.ParameterList([W_txh, W_thh, b_th])


# 定义模型
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


def rnn(inputs, state, params):
    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q, W_txh, W_thh, b_th = params
    H, = state
    outputs = []
    for X in inputs:
        # X : batch_size * vocab_size
        # W_xh : vocab_size * hidden
        # H : t * batch_size * hidden
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
        if(Deep_Layer == 1):
            Y = torch.matmul(H, W_hq) + b_q
        else:
            Y = H
        outputs.append(Y)
    # print(i)

    for i in range(Deep_Layer - 1):
        _W_xh, _W_hh, _b_h, _W_hq, _b_q, W_txh, W_thh, b_th = params
        H, = state
        n_outputs = []
        for H_i in outputs:
            H = torch.tanh(torch.matmul(H_i, W_txh) + torch.matmul(H, W_thh) + b_th)

            if(i < Deep_Layer - 2):
                n_outputs.append(H)
            else:
                Y = torch.matmul(H, W_hq) + b_q
                # print(Y)
                n_outputs.append(Y)
        outputs = n_outputs
    # print(np.array(outputs).shape)

    return outputs, (H,)

num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

d2l.train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, True, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)
