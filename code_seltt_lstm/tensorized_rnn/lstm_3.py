import torch
from torch import nn

from .rnn_utils import ActivGradLogger
from .rnn_utils import param_count as pc

class LSTMCell(nn.Module):
    print("*** LSTMCell ***")
    def __init__(self, input_size, hidden_size, bias, device):
        print("*** <START> LSTMCell <def __init>")
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.input_weights = self._create_input_hidden_weights()
        self.hidden_weights = self._create_hidden_hidden_weights()
        print("*** <END> LSTMCell <def __init>")

    def _create_input_hidden_weights(self):
        print("*** <START> LSTMCell <def _create_input_hidden_weights>")
        print("***  RETURN> nn.Linear")
        tmp1 = nn.Linear(self.input_size, 4 * self.hidden_size, False).to(self.device)
        print("***          : ",tmp1)
        return tmp1

    def _create_hidden_hidden_weights(self):
        print("*** <START> LSTMCell <def _create_hidden_hidden_weights>")
        print("***  RETURN> nn.Linear")
        tmp2 = nn.Linear(self.hidden_size, 4 * self.hidden_size, self.bias).to(self.device)
        print("***         : ",tmp2)
        return tmp2

    def forward(self, input, hx, cx):
        gates_out = self.input_weights(input) + self.hidden_weights(hx)

        ingate = torch.sigmoid(gates_out[:, :self.hidden_size])
        forgetgate = torch.sigmoid(gates_out[:, self.hidden_size:2*self.hidden_size])
        cellgate = torch.tanh(gates_out[:, 2*self.hidden_size:3*self.hidden_size])
        outgate = torch.sigmoid(gates_out[:, 3*self.hidden_size:])

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        # Register gradient hooks if we have them
        if hasattr(self, '_h_backward_hook') and hy.requires_grad:
            assert hasattr(self, '_c_backward_hook')
            assert cy.requires_grad
            hy.register_hook(self._h_backward_hook)
            cy.register_hook(self._c_backward_hook)

        return hy, cy


class LSTM(nn.Module):
    print(" <<< LSTM >>>")
    def __init__(self, input_size, hidden_size, num_layers, num_flayers, device,
                 bias=True):
        print("<START> LSTM <__init__>")
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_flayers = num_flayers
        self.bias = bias
        self.device = device


        # instantiate lstm cell for each layer.
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)       # name: cell0, cell1 ...
            if i == 0:
                cell = self._create_first_layer_cell()
            elif i < self.num_flayers:          # sel에만 걸림
                cell = self._create_other_layer_lstm_cell()
            else:
                cell = self._create_other_layer_cell()
            setattr(self, name, cell)
            self._all_layers.append(cell)

        total = 0
        for cell in self._all_layers:
            print("len(self._all_layers):{}".format(len(self._all_layers)))
            print("cell:{}".format(cell))
            for attr in ('input_weights', 'hidden_weights'):
                total += pc(getattr(cell, attr))  # rnn_util.py의 param_count()불러서 셈
        print("total = {}".format(total))
        print("<END> LSTM <__init__>")

    def _create_first_layer_cell(self):
        print("<START> LSTM <def _create_first_layer_cell>")
        print(" RETRUN> <Call> LSTMCell")
        return LSTMCell(self.input_size, self.hidden_size, self.bias, self.device)

    def _create_other_layer_cell(self):
        print("<START><lstm.py><class LSTM><def _create_other_layer_cell>")
        print(" RETRUN> <Call> LSTMCell")
        return LSTMCell(self.hidden_size, self.hidden_size, self.bias, self.device)

    def init_hidden(self, batch_size):
        print("<START><lstm.py><class LSTM><init_hidden>")
        h = torch.zeros(batch_size, self.hidden_size).to(self.device)
        c = torch.zeros(batch_size, self.hidden_size).to(self.device)
        print(" RETRUN> h, c ")
        return h, c

    def param_count(self):
        print("<START> LSTM <param_count>")
        # input_weights and hidden_weights
        total = 0
        for cell in self._all_layers:
            print(" len(self._all_layers):{}".format(len(self._all_layers)))
            print(" cell.shape:{}".format(cell.shape))
            for attr in ('input_weights', 'hidden_weights'):
                total += pc(getattr(cell, attr))                    #rnn_util.py의 param_count()불러서 셈
        print(" total = {}".format(total))
        print(" Return> total")
        return total

    def forward(self, input, init_states=None):
        # print("<START><lstm.py><Class LSTM><def forward>")
        """
        :param input:       Tensor of input data of shape (batch_size, seq_len, input_size).
        :param init_states: Initial hidden states of LSTM. If None, is initialized to zeros.
                            Shape is (batch_size, hidden_size).

        :return:    outputs, (h, c)
                    outputs:  Torch tensor of shape (seq_len, batch_size, hidden_size) containing
                              output features from last layer of LSTM.
                    h:        Output features (ie hiddens state) from last time step of the last layer.
                              Shape is (batch_size, hidden_size)
                    c:        Cell state from last time step of the last layer.
                              Shape is (batch_size, hidden_size)
        """

        batch_size, seq_len, input_size = input.size()
        print("YJYJ4 :{}".format(input.size()))
        outputs = torch.zeros(batch_size, seq_len, self.hidden_size).to(input.device)

        # initialise hidden and cell states.
        (h, c) = self.init_hidden(batch_size) if init_states is None else init_states
        internal_state = [(h, c)] * self.num_layers

        for step in range(seq_len):
            x = input[:, step, :]
            for i in range(self.num_layers):
                # name = 'cell{}'.format(i)
                # lstm_cell = getattr(self, name)
                lstm_cell = self._all_layers[i]

                (h, c) = internal_state[i]
                x, new_c = lstm_cell(x, h, c)           # 여기서 lstm_cell Forward 콜되었다!!! 윗윗줄에 _all_layers불리면 Cell로 넘어가는 구조
                internal_state[i] = (x, new_c)
            outputs[:, step, :] = x

        # print("<END><lstm.py><Class LSTM><def forward>")
        # print("<RETURN>outputs, (x, new_c)")
        return outputs, (x, new_c)
