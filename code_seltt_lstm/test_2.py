"""
for MNIST DATA
"""
# -*- coding: utf-8 -*-
import os, sys

from time import time
# from comet_logger import CometLogger

import torch
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from torch.autograd import Variable
import warnings


from utils_2 import data_generator, count_model_params
from classifier import MNIST_Classifier




### Running GPU Setting ##
os.environ["CUDA_VISIBLE_DEVICES"] = '4'


##### 1. BASIC SETTING #####
## 1.(1) Argument Setting ##
parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--mode', type=int, default=0,
                    help='use selective tensorized RNN model (default: 0/ 0:basic, 1:tt, 2:selective)')
parser.add_argument('--n_layers', type=int, default=1,
                    help='# of layers (default: 1)')
parser.add_argument('--n_front_layers', type=int, default=1,
                    help='# of front layers (default: 1)')
parser.add_argument('--epochs', type=int, default=2,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--hidden_size', type=int, default=256,
                    help='number of hidden units per layer (default: 256)')

parser.add_argument('--ncores', type=int, default=2,
                    help='number of TT cores (default: 2)')
parser.add_argument('--ttrank', type=int, default=2,
                    help='TT rank (default: 2)')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')

parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--gru', action='store_true',
                    help='use GRU instead of LSTM (default: False)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='initial learning rate (default: 1e-2)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--lr_scheduler', action='store_false',
                    help='Whether to use piecewise-constant LR scheduler '
                         '(default: True)')

parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: False)')
parser.add_argument('--naive_tt', action='store_true',
                    help='use naive tensorized RNN model (default: False).')
# parser.add_argument('--enable_logging', action='store_true',
#                     help='Log metrics to Comet and save model to disk (default: False)')
parser.add_argument('--extra_core', type=str, default='none',
                    help='Where to place extra core, if any (options: none, first, last)')
parser.add_argument("--gpu_no", type=int, default=0, help =\
                "The index of GPU to use if multiple are available. If none, CPU will be used.")

# Set tt ??????????????????
args = parser.parse_args()
if not args.mode:                     #mode??? basic??????  args.ncores, ttrank = 1
    args.ncores = 1
    args.ttrank = 1
assert not (args.naive_tt and not args.mode)      # args.naive_tt = False, args.tt = False
assert args.extra_core in ['none', 'first', 'last']
if args.extra_core == 'none': args.extra_core = None


## 1.(2) fix seeds ##
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)


## 1.(3) Set cuda device ##
if torch.cuda.is_available():
    # set default cuda device.
    device = torch.device('cuda:{}'.format(args.gpu_no))
    torch.cuda.set_device(device)
    print("YJ:cuda is working well")
    # warn if not using cuda and gpu is available.
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
else:
    device = torch.device('cpu')


##  1.(4) Set Data and Parameters ##
root = './data/mnist'
batch_size = args.batch_size
n_classes = 10
epochs = args.epochs
steps = 0
# print("batch size: {}, epochs: {}".format(batch_size,epochs))
if args.permute:    # each pixel is input for a time step.
    input_channels = 1      # [128,1,28,28] ???????????? ????????? ????????????(output = model(data)), [batch_size,seq_len,input_size]=[128,784,1]??? ???????????????
                            # ????????? 784????????? 1??? ???????????? ??????
    # print("args.permuted. input_channels= {}".format(input_channels))
else:               # each row(of pixels) is input for a time step.
    input_channels = 28     # [128,1,28,28] ???????????? ????????? ????????????(output = model(data)), [batch_size,seq_len,input_size]=[128,28,28]??? ???????????????
    # print("NOT args.permuted. input_channels= {}".format(input_channels))
seq_length = int(784 / input_channels)
# print("seq_length(287/input_channels): {}".format(seq_length))
# print("args: ",args)



## 1.(5) Save the data ##
mode_name = 'sel' if args.mode == 2 else ('tt' if args.mode == 1 else 'basic')
gru_name = 'gru' if args.gru else 'lstm'
if args.mode == 2:
    name = (f"{gru_name}_in{input_channels}_{mode_name}_n{args.n_layers}({args.n_front_layers}+{args.n_layers - args.n_front_layers})"
            f"_ncores{args.ncores}_r{args.ttrank}_h{args.hidden_size}_ep{args.epochs}")
else:
    name = (f"{gru_name}_in{input_channels}_{mode_name}_n{args.n_layers}"
        f"_ncores{args.ncores}_r{args.ttrank}_h{args.hidden_size}_ep{args.epochs}")

sys.stdout = open(r'./results/'+ name + '.txt', 'w')  # run?????? ????????? ????????? ??????
print("File name : ",name)
print("Arguments: ",args)
print("Input Data Shape : [batch_size,seq_len,input_size]=[{},{},{}]".format(batch_size,seq_length,input_channels))





##### 2. DATA LOADING #####
print("\n### Data Loaded from data loader ###")
# print("# train_loader: ????????? 128???, 50,000/128(batch) = 390.xx 390??? ???????????? ???????????????")
# print("# val_loader, test_loader????????????")
train_loader, val_loader, test_loader = data_generator(root, batch_size)
print("train_loader: {}, \nval_loader: {}, \ntest_loader: {}".format(train_loader, val_loader, test_loader))




##### 3. Define Model  #####
### 3.(1) Main Model  ###
print("\n### Model ?????? ###")
model = MNIST_Classifier(input_channels, n_classes, args.hidden_size, args.n_layers, args.n_front_layers, device,
                         mode=args.mode, gru=args.gru, n_cores=args.ncores,
                         tt_rank=args.ttrank, naive_tt=args.naive_tt,
                         extra_core=args.extra_core)


n_trainable, n_nontrainable = count_model_params(model)             #utils.py ?????? count_model_params????????????
print("Model instantiated. Trainable params: {}, Non-trainable params: {}. Total: {}"
      .format(n_trainable, n_nontrainable, n_trainable + n_nontrainable))



### 3.(2) ????????????  ### ?????? ????????? ??????????????? ???????????????
# ????????? ?????? ???????????? ???????????? 28*28=784 / ????????? ??????????????? 0~9???????????? 10???.
# np.random.permutation(5)=[4 1 3 0 2]...
# torch.Tensor~.long() => tensor([4, 1, 3, 0, 2])
# ?????? ?????? ???????????? 784?????? ?????????
permute = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()
#??? ??????????????? ??????????????? ????????????
# model.cuda() by default will send your model to the "current device", which can be set with torch.cuda.set_device(device).
# An alternative way to send the model to a specific device is model.to(torch.device('cuda:0'))
if args.cuda:
    model.cuda()
    permute = permute.cuda()
    print("Successfully Sent the model to the current device")


### 3.(3) Set learning rate, optimizer, scheduler ###
lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
if args.lr_scheduler:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

# print(f"input channels: {input_channels}; seq_length: {seq_length}; cuda: {args.cuda}")
# exit()






##### 4. TRAINING #####
def train(ep):
    global steps
    train_loss = 0
    train_correct = 0
    model.train()                                                   # (1) ????????? train????????? ??????
    for batch_idx, (data, target) in enumerate(train_loader):       # ?????????????????? 50,000/128=390.xx 390??? ???????????? ????????????
        # print(batch_idx)
        if args.cuda:
            data, target = data.cuda(), target.cuda()               # (2) ???????????? ???????????? ????????? ??????
        # print("<CHECK1> data : ", data.shape)
        data = data.view(-1, seq_length, input_channels)
        # print("<CHECK2> data : ", data.shape)
        if args.permute:
            data = data[:, permute, :]
        # print("<CHECK3> data : ", data.shape)
        data, target = Variable(data), Variable(target)             # (3) ????????? ??????. Variable():autograd ???????????? ??????: backprop?????? ????????? ?????? ??????
                                                                    #      a=Variable(a, requires_grad=True)??? a.data/a.grad/a.grad_fn 3?????? ?????? ??????
                                                                    #      ?????? ????????? backward()??? ????????? ??????????????? ????????? ????????????
        optimizer.zero_grad()                                       # (4) ?????????????????? ?????????????????? 0?????? ??????????????? ????????????. ??????????????? ??????????????? ??? ??????
        # print("<CHECK4> data: {}".format(data.shape))
        output = model(data)                                        # (5) forward propagation
        # print("<CHECK5> output: {}".format(output.shape))
        # print("         target: {}".format(target.shape))
        loss = F.nll_loss(output, target)                           # (6) loss ??????
        loss.backward()                                             # (7) Back propagation
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()                                            # (8) ????????? ????????????

        pred = output.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        train_loss += loss                                          # train_loss?????? ???????????? loss??? ?????? ???????????????
        # print(f"step: {steps}; loss: {loss}")
        steps += 1
        if batch_idx > 0 and (batch_idx + 1) % args.log_interval == 0:          # ?????????????????? ?????? log_interval=100????????? ???????????????
            avg_train_loss = train_loss.item() / args.log_interval              # 100???????????? ?????? ???????????? ?????? ??? acc
            avg_train_acc = 100. * train_correct.item() / (args.log_interval * batch_size)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: ({:.2f}%)\tSteps: {}'.format(
                ep, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), avg_train_loss, avg_train_acc, steps))
            train_loss = 0
            train_correct = 0





##### 5. TESTING #####
best_test_acc = 0.0
def test(test_model, loader, val_or_test="val"):
    test_model.eval()
    test_loss = 0
    correct = 0
    global best_test_acc
    with torch.no_grad():
        for data, target in loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(-1, seq_length, input_channels)
            if args.permute:
                data = data[:, permute, :]
            data, target = Variable(data), Variable(target)
            output = test_model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(loader.dataset)
        test_acc = 100. * correct / len(loader.dataset)
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            val_or_test, test_loss, correct, len(loader.dataset), test_acc))

        return test_loss





if __name__ == "__main__":
    import os.path
    start = time()

    for epoch in range(1, epochs+1):
        train(epoch)
        test(model, val_loader, "val")
        if args.lr_scheduler: scheduler.step()
        print(f"Runtime: {time() - start:.0f} sec\n")

sys.stdout.close()
