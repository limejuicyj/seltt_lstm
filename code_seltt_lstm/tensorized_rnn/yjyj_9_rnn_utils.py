from collections import deque

import torch

from t3nsor.tensor_train import TensorTrain, TensorTrainBatch
from t3nsor.utils import auto_shape

# from collections import deque
#
# import torch
#
# import sys
# sys.path.append("..")
# from t3nsor.tensor_train import TensorTrain, TensorTrainBatch
# from t3nsor.utils import auto_shape

def tt_shape(in_features, out_features, n_cores, n_gates, new_core=None):
    print("\t<START><rnn_utils.py><DEF tt_shape>")
    print("\tin_feature:{}, out_features:{}, n_cores:{}, n_gates:{}, new_core:{}".format(in_features, out_features, n_cores, n_gates, new_core))
    assert new_core in [None, 'first', 'last']

    if new_core is None:
        out_features *= n_gates                                         # n_gates=4 이면 4^4 256
    print("\tout_features:{}".format(out_features))
    print("\tauto-shape",auto_shape(in_features,d=n_cores))
    in_quant = [int(n) for n in auto_shape(in_features, d=n_cores)]
    out_quant = [int(n) for n in auto_shape(out_features, d=n_cores)]
    print("\tin_quant : {}, out_quant: {}".format(in_quant,out_quant))
    if new_core == 'first':
        in_quant = [1] + in_quant
        out_quant = [n_gates] + out_quant
    elif new_core == 'last':
        in_quant = in_quant + [1]
        out_quant = out_quant + [n_gates]
    print("\tin_quant : {}, out_quant: {}".format(in_quant, out_quant))
    print("\t<END><rnn_utils.py><DEF tt_shape>")
    print("\t>>><RETURN> [in_quant, out_quant]")
    return [in_quant, out_quant]
    # TTLinear(in_features=in_features, out_features=out_features,
    #                             bias=bias, auto_shapes=auto_shapes, d=d, tt_rank=tt_rank,
    #                             init=init, shape=shape, auto_shape_mode=auto_shape_mode,
    #                             auto_shape_criterion=auto_shape_criterion)

class ActivGradLogger():
    # Global info for dealing with different loggers
    all_loggers = dict()

    @staticmethod
    def get_logs():
        log_dict = dict()
        quantities = ['act', 'log_act', 'grad', 'log_grad']
        for var, logger in ActivGradLogger.all_loggers.items():
            for qnt in quantities:
                record_mat = getattr(logger, f'{qnt}_epoch')
                log_dict[(var, qnt)] = torch.stack(record_mat)

        return log_dict

    def __init__(self, name):
        # Name of the variable being logged
        all_loggers = ActivGradLogger.all_loggers
        assert name not in all_loggers
        all_loggers[name] = self
        self.name = name

        # Info for storing the activations and gradients between epochs
        self.act_epoch = []
        self.grad_epoch = []
        self.log_act_epoch = []
        self.log_grad_epoch = []

        # Info for storing the activations and gradients within an epoch
        self.act_mini = []
        self.grad_mini = []
        self.log_act_mini = []
        self.log_grad_mini = []

        # Info for storing the activations and gradients within a minibatch
        self.act = []
        self.grad = deque()
        self.log_act = []
        self.log_grad = deque()
        
    @staticmethod
    def get_logger(name):
        assert isinstance(name, dict)
        all_loggers = ActivGradLogger.all_loggers
        if name in all_loggers:
            return all_loggers[name]
        else:
            print(f"Logger '{name}' not yet initialized")

    @staticmethod
    def end_epoch():
        all_loggers = ActivGradLogger.all_loggers
        for logger in all_loggers.values():
            logger._end_epoch()

    @staticmethod
    def end_minibatch():
        all_loggers = ActivGradLogger.all_loggers
        for logger in all_loggers.values():
            logger._end_minibatch()

    @staticmethod
    def del_record():
        all_loggers = ActivGradLogger.all_loggers
        for logger in all_loggers.values():
            logger._del_record()

    def create_hooks(self, output_ind):
        @torch.no_grad()
        def forward_hook(rnn_cell, inputs, outputs):
            # Get the average and average log of the value of interest
            if not isinstance(outputs, tuple):
                assert isinstance(outputs, torch.Tensor)
                assert output_ind == 0
                outputs = (outputs,)
            target = outputs[output_ind].detach()
            assert isinstance(target, torch.Tensor)
            av_act = av_norm(target)
            av_log_act = av_norm(target, average_logs=True)

            # Add to our running list
            self.act.append(av_act)
            self.log_act.append(av_log_act)

        @torch.no_grad()
        def backward_hook(grad_out):
            target = grad_out.detach()
            assert isinstance(target, torch.Tensor)
            av_grad = av_norm(target)
            av_log_grad = av_norm(target, average_logs=True)

            # Add to our running list (gradients must be prepended, since
            # they're added in reverse order)
            self.grad.appendleft(av_grad)
            self.log_grad.appendleft(av_log_grad)

        return forward_hook, backward_hook

    def _end_epoch(self):
        # Average across minibatches and append to global list
        self.act_epoch.append(torch.mean(torch.stack(self.act_mini), 0))
        self.grad_epoch.append(torch.mean(torch.stack(self.grad_mini), 0))
        self.log_act_epoch.append(torch.mean(torch.stack(self.log_act_mini), 0))
        self.log_grad_epoch.append(torch.mean(torch.stack(self.log_grad_mini), 0))

        # Erase all single-epoch records
        self.act_mini = self.grad_mini = self.log_act_mini = self.log_grad_mini = []

    def _end_minibatch(self):
        if self.act_mini == []:
            assert self.log_act_mini == self.grad_mini == self.log_grad_mini == []
            self.act_mini = [torch.stack(self.act)]
            self.log_act_mini = [torch.stack(self.log_act)]
            self.grad_mini = [torch.stack(list(self.grad))]
            self.log_grad_mini = [torch.stack(list(self.log_grad))]

        else:
            assert len(self.act_mini[0]) == len(self.act)
            assert len(self.grad_mini[0]) == len(self.grad)
            assert len(self.log_act_mini[0]) == len(self.log_act)
            assert len(self.log_grad_mini[0]) == len(self.log_grad)
            self.act_mini.append(torch.stack(self.act))
            self.log_act_mini.append(torch.stack(self.log_act))
            self.grad_mini.append(torch.stack(list(self.grad)))
            self.log_grad_mini.append(torch.stack(list(self.log_grad)))

        # Delete all gradients and activations
        self._del_record()

    def _del_record(self):
        del self.act[:]
        del self.log_act[:]
        for i in range(len(self.grad)-1, -1, -1): del self.grad[i]
        for i in range(len(self.log_grad)-1, -1, -1): del self.log_grad[i]

def av_norm(tensor, average_logs=False):
    num_inds = len(tensor.shape)
    norms = (tensor ** 2).sum(list(range(1, num_inds)))
    if average_logs: norms = torch.log(norms)
    assert len(norms.shape) == 1
    return norms.mean()



def project_ttgrad(base_tt, grad_tt):
    # Unpack inputs to get core lists
    assert all(isinstance(c, TensorTrainBatch) for c in [base_tt, grad_tt])
    base_cores, grad_cores = base_tt.tt_cores, grad_tt.tt_cores

    # Check formatting of tensors contained in base_cores and grad_cores
    assert len(base_cores) == len(grad_cores)
    assert all(b.shape == g.shape for b, g in zip(base_cores, grad_cores))
    assert all(len(c.shape) == 5 for c in base_cores)
    num_cores = len(base_cores)
    batch_size = base_cores[0].shape[0]
    rank_dims = [1] + [b.shape[-1] for b in base_cores]
    assert all(b.shape[0] == batch_size for b in base_cores)

    # Create a TT matrix representing the global gradient, which has a
    # block structure of the form    ->    [[base_core, grad_core],
    #                                                0, base_core]]
    local_tt = []
    for b_core, g_core in zip(base_cores, grad_cores):
        b, l, lm, rm, r = b_core.shape
        # Template for the TT core of local gradient
        local_core = torch.zeros(b, 2*l, lm, rm, 2*r)
        # Diagonal base core terms
        for i in range(2):
            local_core[:, i*l:(i+1)*l, :, :, i*r:(i+1)*r] = b_core
        # Upper right gradient term
        local_core[:, :l, :, :, r:] = g_core
        # Add core to the TT matrix core list
        local_tt.append(local_core)

    # Pick out the correct boundary conditions to get just the gradient
    local_tt[0]  = local_tt[0][:, 0:1, :, :, :]
    local_tt[-1] = local_tt[-1][:,  :, :, :, 1:2]

    # Convert to TensorTrainBatch and use full() method to get global mats
    return TensorTrainBatch(local_tt).full()



# ## Below gives an example usage of project_ttgrad
# if __name__ == '__main__':
#     # Batch size of 10, three cores with TT ranks 5 and 7, 
#     # all three core mat dims are 2, which gives an 8x8 global mat
#     shapes = [[10, 1, 2, 2, 5], [10, 5, 2, 2, 7], [10, 7, 2, 2, 1]]
#     # Cores are chosen as all-ones tensors
#     base_cores = grad_cores = [torch.ones(*s) for s in shapes]
#     base_tt = TensorTrainBatch(base_cores)
#     grad_tt = TensorTrainBatch(grad_cores)

#     # Project gradient TT cores at base TT matrix to get global gradients
#     grad_mats = project_ttgrad(base_tt, grad_tt)

#     print(grad_mats[0]) # First global grad matrix in batch
#     # Grad entries are 105 = 35 * 3 (product of all TT ranks and 
#     #                                number of cores defining TT matrix)

def param_count(matrix):
    """Count the number of weights in a matrix or TT matrix"""
    assert isinstance(matrix, torch.nn.Module)
    total = 0
    for param in matrix.parameters():
        num = param.shape
        total += num.numel()
    if isinstance(matrix, TensorTrain):
        assert total == matrix.dof
    
    return total