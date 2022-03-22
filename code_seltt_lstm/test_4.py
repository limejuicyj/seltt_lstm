"""
for ELE DATA
"""
# -*- coding: utf-8 -*-
import os, sys

from time import time
from comet_logger import CometLogger

import torch
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from torch.autograd import Variable
import warnings


from utils_4 import data_generator, count_model_params, Data_Type
from classifier_4 import MNIST_Classifier


### Running GPU Setting ##
os.environ["CUDA_VISIBLE_DEVICES"] = '4'


##### 1. BASIC SETTING #####
## 1.(1) Argument Setting ##
parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--mode', type=int, default=2,
                    help='use selective tensorized RNN model (default: 0/ 0:basic, 1:tt, 2:selective)')
parser.add_argument('--n_layers', type=int, default=3,
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

parser.add_argument('--year', type=int, default=1,
                    help='set dataset to use (default: 1)')

# Set tt 안걸리는경우
args = parser.parse_args()
if not args.mode:                     #mode가 basic이면  args.ncores, ttrank = 1
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

root = r'./ttlstm_commit/'
filename = 'data/space/altitude/K3A_EPH_ELE_altitude.npy'

batch_size = args.batch_size
n_classes = 6
epochs = args.epochs
steps = 0
if args.permute:    # each pixel(R,G,B pairs) is input for a time step.
    input_channels = 1      # input shape: [batch_size,seq_len,input_size]=[128, 30, 1]
else:               # each row(of pixels) is input for a time step.
    input_channels = 6     # input shape: [batch_size,seq_len,input_size]=[128, 5, 6]
x_seq_length = 5 # 현재: 궤도 6요소 -> 궤도 6요소 예측 / 궤도 6요소 -> altitude 예측할 때 다시 생각해봐야 함.
y_seq_length = 1
data_type = args.year
year = ['full', '2016', '2017', '2018', '2019', '2020'][data_type]


## 1.(5) Save the data ##
mode_name = 'sel' if args.mode == 2 else ('tt' if args.mode == 1 else 'basic')
gru_name = 'gru' if args.gru else 'lstm'
if args.mode == 2:
    name = (f"{gru_name}_in{input_channels}_{mode_name}_n{args.n_layers}({args.n_front_layers}+{args.n_layers - args.n_front_layers})"
            f"_ncores{args.ncores}_r{args.ttrank}_h{args.hidden_size}_x{x_seq_length}_y{y_seq_length}_{year}_lr{args.lr}_ep{args.epochs}")
else:
    name = (f"{gru_name}_in{input_channels}_{mode_name}_n{args.n_layers}"
        f"_ncores{args.ncores}_r{args.ttrank}_h{args.hidden_size}_x{x_seq_length}_y{y_seq_length}_{year}_lr{args.lr}_ep{args.epochs}")

sys.stdout = open(root + 'results/ele_'+ name + '.txt', 'w')  # run결과 외부에 파일로 저장
print("File name : ",name)
print("Arguments: ",args)
print("Input Data Shape : [batch_size,seq_len,input_size]=[{},{},{}]".format(batch_size,x_seq_length,input_channels))





##### 2. DATA LOADING #####
print("\n### Data Loaded from data loader ###")
train_loader, val_loader, test_loader = data_generator(root, filename, x_seq_length, y_seq_length, batch_size, data_type)
print("train_loader: {}, \nval_loader: {}, \ntest_loader: {}".format(train_loader, val_loader, test_loader))





##### 3. Define Model  #####
### 3.(1) Main Model  ###
print("\n### Model 정의 ###")
model = MNIST_Classifier(input_channels, n_classes, args.hidden_size, args.n_layers, args.n_front_layers, device,
                         x_seq_length, y_seq_length,
                         mode=args.mode, gru=args.gru, n_cores=args.ncores,
                         tt_rank=args.ttrank, naive_tt=args.naive_tt,
                         extra_core=args.extra_core)


n_trainable, n_nontrainable = count_model_params(model)             #utils.py 아래 count_model_params함수있음
print("Model instantiated. Trainable params: {}, Non-trainable params: {}. Total: {}"
      .format(n_trainable, n_nontrainable, n_trainable + n_nontrainable))




### 3.(2) 입력벡터  ### 이거 퍼뮤트 트루일때만 해당하는듯
# 모델에 넣을 입력벡터 사이즈는 28*28=784 / 참고로 출력벡터는 0~9까지니까 10임.
# np.random.permutation(5)=[4 1 3 0 2]...
# torch.Tensor~.long() => tensor([4, 1, 3, 0, 2])
# 결국 그냥 한줄짜리 784길이 정수들
permute = torch.Tensor(np.random.permutation(1024).astype(np.float64)).long()
#이 입력벡터를 쿠다에다가 던져준다
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

loss_fn = torch.nn.MSELoss()


##### 4. TRAINING #####
def train(ep):
    global steps
    train_loss = 0
    train_correct = 0
    model.train()                                                   # (1) 모델을 train모드로 변환
    for batch_idx, (data, target) in enumerate(train_loader):       # 데이터로더가 50,000/128=390.xx 390번 데이터를 로드해줌
        # print(batch_idx)
        if args.cuda:
            data, target = data.cuda(), target.cuda()               # (2) 데이터랑 라벨이랑 쿠다로 보냄
        # print("<CHECK1> data : ",data.shape)
        data = data.view(-1, x_seq_length, input_channels)
        # data = data.view(batch_size, seq_length, input_channels)
        # print("<CHECK2> data : ", data.shape)
        if args.permute:
            data = data[:, permute, :]
        # print("<CHECK3> data : ", data.shape)
        data, target = Variable(data), Variable(target)             # (3) 미분값 계산. Variable():autograd 안에있는 함수: backprop위한 미분값 자동 계산
                                                                    #      a=Variable(a, requires_grad=True)면 a.data/a.grad/a.grad_fn 3개의 값을 가짐
                                                                    #      후에 반드시 backward()를 써줘야 자동계산한 값들이 반영이됨
        optimizer.zero_grad()                                       # (4) 옵티마이저의 그래디언트를 0으로 초기화한번 해줘야함. 그렇지않음 버퍼걸려서 잘 안됨
        # print("<CHECK4> data: {}".format(data.shape))
        output = model(data)                                        # (5) forward propagation
        #print("check2>> output: {}".format(output.shape))
        #print("      >> target: {}".format(target.shape))
        loss = loss_fn(output, target)                           # (6) loss 계산
        loss.backward()                                             # (7) Back propagation
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()                                            # (8) 가중치 업데이트

        #pred = output.data.max(1, keepdim=True)[1]
        #train_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        train_loss += loss                                          # train_loss에는 그동안의 loss를 계속 더해놓는다
        # print(f"step: {steps}; loss: {loss}")
        steps += 1
        if batch_idx > 0 and (batch_idx + 1) % args.log_interval == 0:          # 트레이닝상황 출력 log_interval=100번마다 한번씩하기
            avg_train_loss = train_loss.item() / args.log_interval              # 100번동안의 평균 트레이닝 로스 및 acc
            #avg_train_acc = 100. * train_correct.item() / (args.log_interval * batch_size)
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: ({:.2f}%)\tSteps: {}'.format(
            #     ep, batch_idx * batch_size, len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), avg_train_loss, avg_train_acc, steps))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                ep, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), avg_train_loss, steps))
            train_loss = 0
            train_correct = 0





##### 5. TESTING #####
best_test_acc = 0.0
separate_file_for_accuracy = 1
def test(test_model, loader, epoch, val_or_test="val"):
    test_model.eval()
    test_loss = 0
    correct = 0
    global best_test_acc
    with torch.no_grad():
        for data, target in loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(-1, x_seq_length, input_channels)
            if args.permute:
                data = data[:, permute, :]
            data, target = Variable(data), Variable(target)
            output = test_model(data)
            test_loss += loss_fn(output, target).item()
            #pred = output.data.max(1, keepdim=True)[1]
            #correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(loader.dataset)
        #test_acc = 100. * correct / len(loader.dataset)
        #print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        #   val_or_test, test_loss, correct, len(loader.dataset), test_acc))

        return test_loss





if __name__ == "__main__":
    import os.path
    start = time()

    for epoch in range(1, epochs+1):
        train(epoch)
        test_loss = test(model, val_loader, epoch, "val")
        if args.lr_scheduler: scheduler.step()
        if separate_file_for_accuracy: 
            with open(root + 'results/ele_'+ name + '_acc.txt', 'a') as f:
                f.write('\n{} set: Epoch: {} Average loss: {:.8f}, Runtime: {:.0f} sec\n'.format(
                    "val", epoch, test_loss, time() - start))
        else:
            print('\n{} set: Average loss: {:.8f} \n'.format("val", test_loss))
            print(f"Runtime: {time() - start:.0f} sec\n")

sys.stdout.close()
