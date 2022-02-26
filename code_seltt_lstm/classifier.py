import torch
import torch.nn.functional as F
from torch import nn

from tensorized_rnn.lstm_2 import LSTM
from tensorized_rnn.tt_lstm_2 import TTLSTM
from tensorized_rnn.seltt_lstm_2 import SelTTLSTM
from tensorized_rnn.gru import GRU, TTGRU
from tensorized_rnn.rnn_utils import param_count as pc
from t3nsor.layers import TTLinear



class MNIST_Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, num_flayers, device,
                 mode=2, gru=True, n_cores=3, tt_rank=2,
                 naive_tt=False, extra_core=None):
        super(MNIST_Classifier, self).__init__()
        self.gru = gru

        print("# input_size: {}".format(input_size))
        print("# output_size: {}".format(output_size))
        print("# hidden_size: {}".format(hidden_size))
        print("# num_layers: {}".format(num_layers))
        print("# num_flayers: {}".format(num_flayers))
        print("# mode: {}".format(mode))
        print("# device: {}".format(device))
        print("# n_cores: {}".format(n_cores))
        print("# tt_rank: {}".format(tt_rank))
        print("# is_naive: {}".format(naive_tt))
        print("# new core: {}".format(extra_core))

        if mode == 2:
            if not gru:     # Seltt_LSTM
                print("\n### Model : Selective TT LSTM    ( 1.SelTTLSTM + 2.TTLinear ) ###")
                print("### 1. CALL SelTTLSTM ###")
                self.rnn = SelTTLSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, num_flayers=num_flayers, device=device,
                                n_cores=n_cores, tt_rank=tt_rank,
                                is_naive=naive_tt,
                                new_core=extra_core)
            else:       # Seltt_GRU
                print("\n### Model : Selective TT GRU    ( 1.SelTTGRU + 2.TTLinear ) ###")
                print("### 1. CALL SelTTGRU ###")
                self.rnn = TTGRU(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, device=device,
                                n_cores=n_cores, tt_rank=tt_rank,
                                is_naive=naive_tt,
                                new_core=extra_core)

            print("\n### 2. CALL TTLinear ###")
            print("\n    in_features=hidden_size:{}, out_features=output_size:{}, bias : True, auto_shapes: True, d=n_cores:{}, "
                  "tt_rank=tt_rank:{}".format(hidden_size, output_size, n_cores, tt_rank))
            self.linear = TTLinear(in_features=hidden_size,
                                   out_features=output_size, bias=True,
                                   auto_shapes=True, d=n_cores, tt_rank=tt_rank)
        elif mode == 1:
            if not gru:     # TT_LSTM
                print("\n### Model : TT LSTM    ( 1.TTLSTM + 2.TTLinear ) ###")
                print("### 1. CALL TTLSTM ###")
                self.rnn = TTLSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, num_flayers=1, device=device,
                                n_cores=n_cores, tt_rank=tt_rank, 
                                is_naive=naive_tt,
                                new_core=extra_core)
            else:       # TT_RGU
                print("\n### Model : TT GRU    ( 1.TTGRU + 2.TTLinear ) ###")
                print("### 1. CALL TTGRU ###")
                self.rnn = TTGRU(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, num_flayers=1, device=device,
                                n_cores=n_cores, tt_rank=tt_rank, 
                                is_naive=naive_tt,
                                new_core=extra_core)

            print("\n### 2. CALL TTLinear ###")
            print("\n    in_features=hidden_size:{}, out_features=output_size:{}, bias : True, auto_shapes: True, d=n_cores:{}, "
                  "tt_rank=tt_rank:{}".format(hidden_size, output_size, n_cores, tt_rank))
            self.linear = TTLinear(in_features=hidden_size, 
                                   out_features=output_size, bias=True, 
                                   auto_shapes=True, d=n_cores, tt_rank=tt_rank)
        else:
            if not gru:     # basic LSTM
                print("\n### Model : Basic LSTM    ( 1.LSTM + 2.Linear ) ###")
                print("### 1. CALL basic LSTM ###")
                self.rnn = LSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, num_flayers=1, device=device)
            else:       # basic GRU
                print("\n### Model : Basic GRU    ( 1.GRU + 2.Linear ) ###")
                print("### 1. CALL basic GRU ###")
                self.rnn = GRU(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, num_flayers=1, device=device)
            print("\n    in_features=hidden_size:{}, out_features=output_size:{}".format(hidden_size, output_size))
            print("### 2. CALL Linear ###")
            self.linear = nn.Linear(hidden_size, output_size)

        param_last_linear = pc(self.linear)
        print("param_last_linear:{}".format(param_last_linear))

    def param_count(self):
        # return self.rnn.param_count() + pc(self.linear)
        param_front = self.rnn.param_count()
        param_back = pc(self.linear)
        print("param_front:{}, param_back:{}".format(param_front, param_back))
        return param_front + param_back

    def forward(self, inputs):
        if self.gru:
            out, last_hidden = self.rnn(inputs)
        else:
            out, (last_hidden, last_cell) = self.rnn(inputs)
            # print("inputs: {}".format(inputs.shape))
            # print("out: {}, out[:,-1,:]: {}".format(out.shape,out[:, -1, :].shape))
            # print("last_hidden: {}, last_cell: {}".format(last_hidden.shape, last_cell.shape))
            # print("last_hidden = last_cell : {}".format(torch.allclose(last_hidden,last_cell)))
        o = self.linear(out[:, -1, :])
        # print("o:{}".format(o.shape))

        return F.log_softmax(o, dim=1)
