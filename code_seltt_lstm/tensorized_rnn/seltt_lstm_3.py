from t3nsor.layers import TTLinear
from .lstm_3 import LSTMCell, LSTM
from .tt_lstm_3 import TTLSTMCell, TTLSTM
from .tt_linearset import TTLinearSet
from .rnn_utils import tt_shape
import torch
from torch import nn

# import sys
# sys.path.append("..")
# from t3nsor.layers import TTLinear
# from tensorized_rnn.lstm import LSTMCell, LSTM
# from tensorized_rnn.tt_linearset import TTLinearSet
# from tensorized_rnn.rnn_utils import tt_shape

class SelTTLSTMCell(LSTMCell):
    print("$$$ SelTTLSTM Cell $$$")
    def __init__(self, input_size, hidden_size, bias, device,           # 이거는 init에서 정의 안하고 아래 각 함수에서 정의하는걸로
                 n_cores, tt_rank, is_naive=False, new_core=None):      # 이거는 init에서 정의
        print("$$$ (1) <START> SelTTLSTM Cell <def _init__> ")
        self.n_cores = n_cores
        self.tt_rank = tt_rank
        self.is_naive = is_naive
        self.new_core = new_core
        self.n_gate = 4
        super().__init__(input_size, hidden_size, bias, device)         # init에서 정의안했으니 여기다시 아규먼트로 들어감
        print("$$$ (1)<END> SelTTLSTM Cell <def __init__>")

    def _create_input_hidden_weights(self):
        print("$$$ (2) <START> SelTTLSTM Cell <def _create_input_hidden_weight>")

        if self.is_naive:
            return TTLinearSet(in_features=self.input_size,
                            out_features=self.hidden_size, n_gates=self.n_gate,
                            bias=False, auto_shapes=True,
                            d=self.n_cores, tt_rank=self.tt_rank).to(self.device)
        else:
            print("$$$ <CALL><rnn_utils.py><DEF tt_shape>")
            shape = tt_shape(self.input_size, self.hidden_size,                     # [[4,7].[32.32]]들어가게됨
                             self.n_cores, self.n_gate, new_core=self.new_core)
            print("$$$ <CALL Back> shape = tt_shape(): {}".format(shape))
            print("$$$ (2) RETURN> TTLinear()")
            return TTLinear(out_features=self.n_gate*self.hidden_size, shape=shape,
                            bias=self.bias, auto_shapes=False, d=self.n_cores,
                            tt_rank=self.tt_rank).to(self.device)

    def _create_hidden_hidden_weights(self):
        print("$$$ (3) <START> SelTTLSTM Cell <def _create_hidden_hidden_weight>")

        if self.is_naive:
            return TTLinearSet(in_features=self.hidden_size,
                            out_features=self.hidden_size, n_gates=self.n_gate,
                            bias=False, auto_shapes=True,
                            d=self.n_cores, tt_rank=self.tt_rank).to(self.device)
        else:
            print("$$$ <CALL><rnn_utils.py><DEF tt_shape>")
            shape = tt_shape(self.hidden_size, self.hidden_size,
                             self.n_cores, self.n_gate, new_core=self.new_core)
            print("$$$ <CALL Back> shape = tt_shape(): {}".format(shape))
            print("$$$ RETURN> TTLinear()")
            aaaa = TTLinear(out_features=self.n_gate*self.hidden_size, shape=shape,
                            bias=self.bias, auto_shapes=False, d=self.n_cores,
                            tt_rank=self.tt_rank).to(self.device)
            print("$$$ aaaa: ",aaaa)
            return aaaa



class SelTTLSTM(LSTM):
    print("<<< SelTTLSTM >>>")
    def __init__(self, input_size, hidden_size, num_layers, num_flayers, device, n_cores,
                 tt_rank, bias=True, is_naive=False, new_core=None):
        print("<START> SelTTLSTM <def __init__>")
        assert new_core in [None, 'first', 'last']
        self.n_cores = n_cores
        self.tt_rank = tt_rank
        self.is_naive = is_naive
        self.new_core = new_core
        super().__init__(input_size, hidden_size, num_layers, num_flayers, device, bias)
        print(" input_size: {}".format(input_size))
        print("<END> TTLSTM <def __init__>")

    def _create_first_layer_cell(self):
        print("<START> SelTTLSTM <def _create_first_layer_cell>")
        print(" RETRUN> LSTMCell(I-H)")
        return LSTMCell(self.input_size, self.hidden_size, self.bias, self.device)

    def _create_other_layer_lstm_cell(self):
        print("<START>  SelTTLSTM  <def _create_other_layer_cell")
        print(" RETRUN> LSTMCell(H-H)")
        return LSTMCell(self.hidden_size, self.hidden_size, self.bias, self.device)

    def _create_other_layer_cell(self):
        print("<START>  SelTTLSTM  <def _create_other_layer_cell")
        print(" RETRUN> SelTTLSTMCell(H-H)")
        return SelTTLSTMCell(self.hidden_size, self.hidden_size, self.bias, self.device,
                          n_cores=self.n_cores, tt_rank=self.tt_rank,
                          is_naive=self.is_naive, new_core=self.new_core)

