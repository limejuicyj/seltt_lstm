"""
K3A_EPH_ELE_altitude.csv 등의 ELE데이터(궤도6요소) 훈련데이터로 넣기위해 정리해서 저장.
(1) 6요소를 훈련해서 그 다음 6요소를 예측 : 데이터에 6요소 들어가고 라벨에 그다음 6요소 저장
(2) 6요소를 훈련해서 altitude를 예측 : 데이터에 6요소 들어가고 라벨에 altitude 저장
"""
import os
import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
import matplotlib.pyplot as plt



# root = './data/mnist'
# transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# train_set = datasets.MNIST(root=root, train=True,
#                              download=True, transform=transform)
# print(" len(train_set) : ", len(train_set))  # 50,000                    [ 50,000, torch.Size[[3, 32, 32] , (int)6] ]
# print(" len(train_set[0]) : ", len(train_set[0]))  # 2
# print(" train_set[0][0].shape : ", train_set[0][0].shape)  # torch.Size([3, 32, 32])
# print(" train_set[0][1] : ", train_set[0][1])  # int / 6 /
#
# one_image, label = train_set[0]
# print("type of one image", type(one_image))
# print("size of one image : ", one_image.shape)
# print("type of label : ", type(label))
# print("label : ", label)
# f = plt.figure()
#
# plt.imshow(one_image.squeeze().numpy(), cmap='gray')
# plt.savefig('test.png')     #서버안에 해당주소가보면 생성되어있음
# plt.show()



# 현재위치 확인
print(os.path.abspath(__file__))
path = os.getcwd()
print(path)

# 파일정보
INPUT_FOLDER = r'/home/youjin/research/tensor_new/data/space/altitude/'
filename = 'K3A_EPH_ELE_altitude.csv'
print(INPUT_FOLDER+filename)


# 파일읽고 필요한 부분만 정리
# data = pd.read_csv(INPUT_FOLDER+filename, header=0)
# # data = data.head(20)              # 실험용으로 몇개만 써볼때 활성화
# print(" data.shape: {}".format(data.shape))
# print(" data.head(10): {}".format(data.head(5)))
# data = data.iloc[:,[2,3,4,5,6,7,10]]
# print(" data.head(10): {}".format(data.head(5)))

# npy 파일로 저장
# np.save(INPUT_FOLDER+filename.split('.')[0],np.array(data))



# npy 파일 로딩 및 확인
data = np.load(INPUT_FOLDER+filename.split('.')[0]+".npy")
np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # 칼럼쪽 생략없이 다 보기
print("data : {}".format(data.shape))
data = data[0:100,:]
print(data.shape)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data = scaler.fit_transform(data)
print(data.shape)
seq_length = 5

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available')

def seq_data(data, seq_length):             # x에는 0~4까지 5개 row가 들어가고, y에 5 (6번째) row 들어감 5일값으로 6일예측
    x_seq = []
    y_seq = []
    for i in range(len(data) - seq_length):
        x_seq.append(data[i: i + seq_length, :])
        y_seq.append(data[i + seq_length, :])

    # return torch.FloatTensor(x_seq).to(device), torch.FloatTensor(y_seq).to(device).view([-1, 1])       # -1은 나머지 하나에따라 결정되는 값. view는 원본바뀌면 같이 바뀜
    return torch.FloatTensor(x_seq).to(device), torch.FloatTensor(y_seq).to(device)       # -1은 나머지 하나에따라 결정되는 값. view는 원본바뀌면 같이 바뀜


# split = 2000000
split = 20
sequence_length = 5

x_seq, y_seq = seq_data(data, sequence_length)
print(x_seq.size(), y_seq.size())
x_train_seq = x_seq[:split]
y_train_seq = y_seq[:split]
x_test_seq = x_seq[split:]
y_test_seq = y_seq[split:]
print(x_train_seq.size(), y_train_seq.size())
print(x_test_seq.size(), y_test_seq.size())

train = torch.utils.data.TensorDataset(x_train_seq, y_train_seq)
test = torch.utils.data.TensorDataset(x_test_seq, y_test_seq)

batch_size = 20
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

input_size = x_seq.size(2)
num_layers = 2
hidden_size = 8



class VanillaRNN(nn.Module):

  def __init__(self, input_size, hidden_size, sequence_length, num_layers, device):
    super(VanillaRNN, self).__init__()
    self.device = device
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Sequential(nn.Linear(hidden_size * sequence_length, 1), nn.Sigmoid())

  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device) # 초기 hidden state 설정하기.
    out, _ = self.rnn(x, h0) # out: RNN의 마지막 레이어로부터 나온 output feature 를 반환한다. hn: hidden state를 반환한다.
    out = out.reshape(out.shape[0], -1) # many to many 전략
    out = self.fc(out)
    return out

model = VanillaRNN(input_size=input_size,
                   hidden_size=hidden_size,
                   sequence_length=sequence_length,
                   num_layers=num_layers,
                   device=device).to(device)

criterion = nn.MSELoss()

lr = 1e-3
num_epochs = 200
optimizer = optim.Adam(model.parameters(), lr=lr)


loss_graph = [] # 그래프 그릴 목적인 loss.
n = len(train_loader)

for epoch in range(num_epochs):
  running_loss = 0.0

  for batch_idx, (data, target) in enumerate(train_loader):

    # seq, target = data # 배치 데이터.
    print("YJYJ : data : ", data.shape)          # torch.Size([20, 5, 7])
    print("YJYJ : target : ", target.shape)         # torch.Size([20, 7])
    data = data.view(-1, sequence_length, input_size)
    out = model(data)   # 모델에 넣고,
    loss = criterion(out, target) # output 가지고 loss 구하고,

    optimizer.zero_grad() #
    loss.backward() # loss가 최소가 되게하는
    optimizer.step() # 가중치 업데이트 해주고,
    running_loss += loss.item() # 한 배치의 loss 더해주고,

  loss_graph.append(running_loss / n) # 한 epoch에 모든 배치들에 대한 평균 loss 리스트에 담고,
  if epoch % 100 == 0:
    print('[epoch: %d] loss: %.4f'%(epoch, running_loss/n))

plt.figure(figsize=(20,10))
plt.plot(loss_graph)
plt.show()


# 하단은 애초에 트레이닝데이터라고 라벨하고 분리해서 저장하는 방식
# pd.set_option('display.max_columns', None)  # 칼럼쪽 생략없이 다 보기
# print(" data.shape: {}".format(data.shape))
# print(" data.head(): {}".format(data.head(20)))
# print(" data.columns.tolist(): {}".format(data.columns.tolist()))
# print(" len(data):{}".format(len(data)))
# print(" data.iloc[0,0]:{}".format(data.iloc[0:2,0:5]))
#
#
# window_size = 4
# # feature_list = []
# # label_list = []
# data_list = []
# for i in range(len(data) - window_size):
#     feature = np.array(data.iloc[i : i + window_size])
#     label = np.array(data.iloc[i + window_size])
#     # feature_list.append(feature)
#     # label_list.append(label)
#     data_list.append([feature,label])
# # return np.array(feature_list), np.array(label_list)
#
#
# # print("feature_list.shape: ",len(feature_list))
# # print("label_list.shape: ",len(label_list))
# # print("feature_list.shape: ",feature_list[0].shape)
# # print("label_list.shape: ",label_list[0].shape)
# # print("feature_list.shape: ",feature_list[1].shape)
# # print("label_list.shape: ",label_list[1].shape)
# print("\n f_l_list : ",len(data_list))
# print("\n f_l_list : ",len(data_list[0]))
# print("\n f_l_list : ",data_list[0][0].shape)
# print("\n f_l_list : ",data_list[0][1].shape)
# # print("\n f_l_list : ",f_l_list[0].shape)
# #
# # # 지금 데이터갯수 20개만으로 되어있고.. 위에 dim틀린거 최종사이즈만 확인하고 저장해서 사용할것.