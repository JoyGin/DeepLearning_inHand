import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

import sys
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('使用的是', device)


# 生成参数
def getparams():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)

    def _three():
        return (
            _one((num_inputs, num_hiddens)),
            _one((num_hiddens, num_hiddens)),
            torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float32),requires_grad=True)
        )
    # input forget output
    W_xi, W_hi, b_i = _three()
    W_xf, W_hf, b_f = _three()
    W_xo, W_ho, b_o = _three()

    # 候选Ct
    W_xc, W_hc, b_c = _three()

    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, dtype=torch.float32), requires_grad=True)

    return torch.nn.ParameterList([W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q])


# 设计初始化H_t
def init_LSTM_state(batch_sizes, num_hiddens, device):
    return (torch.zeros((batch_sizes, num_hiddens), device=device),
            torch.zeros((batch_sizes, num_hiddens), device=device)
            )


def LSTM(inputs, state, params):

    W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = params
    (H, C) = state
    outputs = []

    # inputs : t * batch_size * vocab_size
    # X : batch_size * vocab_size
    # H : batch_size * hidden
    # 两个W分别是：vocab_size * hidden , hidden * hidden
    for X in inputs:
        # 相乘之后 bat_size * hidden
        I_t = torch.sigmoid(torch.matmul(X, W_xi) + torch.matmul(H, W_hi) + b_i)
        F_t = torch.sigmoid(torch.matmul(X, W_xf) + torch.matmul(H, W_hf) + b_f)
        O_t = torch.sigmoid(torch.matmul(X, W_xo) + torch.matmul(H, W_ho) + b_o)
        C_t = torch.tanh(torch.matmul(X, W_xc) + torch.matmul(H, W_hc) + b_c)

        C = F_t * C + I_t * C_t
        H = O_t * torch.tanh(C)

        Y = torch.matmul(H, W_hq) + b_q

        outputs.append(Y)
    return outputs, (H,C)

num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

d2l.train_and_predict_rnn(LSTM, getparams, init_LSTM_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, False, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)