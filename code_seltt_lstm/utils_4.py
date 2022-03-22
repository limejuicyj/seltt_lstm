import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split
from sklearn.preprocessing import MinMaxScaler
from enum import Enum

# definition
data_composition = 1 # 1: (x-궤도 6 요소/y-궤도6요소) 2: (x-궤도6요소/y-altitude) 

year_2016_start = 0
year_2016_dec_start = 482400
year_2016_end = 527041

year_2017_start = 527041
year_2017_dec_start = 1008001
year_2017_end = 1052641

year_2018_start = 1052641
year_2018_dec_start = 1533601
year_2018_end = 1578241

year_2019_start = 1578241
year_2019_dec_start = 2059201
year_2019_end = 2103841

year_2020_start = 2103841
year_2020_dec_start = 2586241
year_2020_end = -1

class Data_Type(Enum):
    FULL = 0
    YEAR_2016 = 1
    YEAR_2017 = 2
    YEAR_2018 = 3
    YEAR_2019 = 4
    YEAR_2020 = 5


class ELEDataset(Dataset): 
    def __init__(self, seq):
        self.x = []
        self.y = []
        for ele in seq:
            self.x.append(ele[0])
            self.y.append(ele[1])

    # 총 데이터의 개수를 리턴
    def __len__(self): 
        return len(self.x)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x[idx])
        y = torch.FloatTensor(self.y[idx])
        return x, y

def define_dataset(data, data_type):
    if data_type == Data_Type.FULL:
        data = data[:, :-1]  
    elif data_type == Data_Type.YEAR_2016:
        data = data[year_2016_start:year_2016_end, :-1] 
    elif data_type == Data_Type.YEAR_2017:
        data = data[year_2017_start:year_2017_end, :-1]
    elif data_type == Data_Type.YEAR_2018:
        data = data[year_2018_start:year_2018_end, :-1]
    elif data_type == Data_Type.YEAR_2019:
        data = data[year_2019_start:year_2019_end, :-1]
    else:
        assert(data_type == Data_Type.YEAR_2020)
        data = data[year_2020_start:year_2020_end, :-1]
    return data

def seq_data(data, x_seq_length, y_seq_length):  # x에는 0~4까지 5개 row가 들어가고, y에 5 (6번째) row 들어감 5일값으로 6일예측
    seq = []
    for i in range(len(data) - (x_seq_length + y_seq_length - 1)): 
        ele = [] 
        ele.append(data[i: i + x_seq_length, :])
        if y_seq_length == 1:
            ele.append(data[i + x_seq_length, :])
        else:
            ele.append(data[i + x_seq_length : i + x_seq_length + y_seq_length, :])
        seq.append(ele) 

    return seq

def split_dataset(seq, data_type):
    if data_type == Data_Type.FULL:
        train_test_split = year_2020_start 
    elif data_type == Data_Type.YEAR_2016:
        train_test_split = year_2016_dec_start
    elif data_type == Data_Type.YEAR_2017:
        train_test_split = year_2017_dec_start
    elif data_type == Data_Type.YEAR_2018:
        train_test_split = year_2018_dec_start
    elif data_type == Data_Type.YEAR_2019:
        train_test_split = year_2019_dec_start
    else:
        assert(data_type == Data_Type.YEAR_2020)
        train_test_split = year_2020_dec_start

    train_seq = seq[:train_test_split]
    test_seq = seq[train_test_split:]
    val_seq = test_seq # val set = test set
    print(len(train_seq), len(val_seq), len(test_seq))

    return  train_seq, val_seq, test_seq

def data_generator(root, filename, x_seq_length, y_seq_length, batch_size, data_type):
    print(root + filename)
    data = np.load(root + filename)
    print("data : {}".format(data.shape))

    # 스케일링
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = define_dataset(data, data_type)
    seq = seq_data(data, x_seq_length, y_seq_length) # seq_data 밑으로 data 사용하면 안됨. 
    print(len(seq), len(seq[0]), len(seq[0][0]), len(seq[0][0][0]))   # [2630877, 2, 5, 6]

    # split train set, val set, and test set
    train_seq, val_seq, test_seq = split_dataset(seq, data_type)

    train_set = ELEDataset(train_seq) 
    val_set = ELEDataset(val_seq)
    test_set = ELEDataset(test_seq)
    print(len(train_set), len(val_set), len(test_set)) 
    
    num_workers=4
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def count_model_params(model):
    """
    Returns number of trainable and non-trainable parameters in a model.
    :param model: A PyTorch nn.Module object.
    :return: A tuple (train_params_count, non_train_params_count)
    """
    print("<START><utils.py><def count_model_params>")
    print("model:{}".format(model))
    train_params_count = 0
    non_train_params_count = 0
    for p in model.parameters():                    #parameters(): 모든 파라미터를 하나씩 반환
        if p.requires_grad:
            train_params_count += p.numel()         # numel(): 원소 갯수
        else:
            non_train_params_count += p.numel()
    print("<END><utils.py><def count_model_params>")
    return train_params_count, non_train_params_count

