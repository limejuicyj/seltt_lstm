import torch
import numpy as np
import tensorly as tl

# from tensorly.decomposition import parafac, matrix_product_state, tensor_train
# from tensor_train import TensorTrain
# from .decompositions import to_tt_tensor
# import t3nsor as t3
from scipy.stats import entropy
import sys
sys.path.append("..")
from t3nsor.decompositions import to_tt_tensor
from t3nsor.utils import _get_all_factors



##### 1. _get_all_factors 함수 확인해보기 #####
a=1024
factors=_get_all_factors(a, d=2, mode='ascending')
print(factors)

##### 2. entropy()함수 확인해보기
weights = [entropy(f) for f in factors]
print(weights)
#[(4, 7), (2, 14)]들어가서 [0.6554817739013927, 0.37677016125643675]가 나옴. (4,7)이 더 값진정보란 소린데 좌우지간 둘중에 하나 선택할라고 만든듯.
i = np.argmax(weights) # [0.6554817739013927, 0.37677016125643675] 이거 둘 중 큰 값의 인덱스반환. 0
print(i)
print(list(factors[i]))

##### 3. to_tt_tensor 함수 확인해보기 ######
dims = (4,6,8)
data = np.random.rand(dims[0]*dims[1]*dims[2])
# data = np.arange(1,9)
data = data.reshape(dims)      # 차원 3차원으로 변경
print("\n### element가 {}개이고 각 차원이 {}인 {}차원 텐서를 쪼개본다 ###".format(data.size, data.shape, data.ndim))
print("\n1. Original data \n{}".format(data.shape))
ft = torch.FloatTensor(data)
print("ft.numel:{}".format(ft.numel()))
print(ft.shape)

tt_shape = to_tt_tensor(ft)

print("=============")

dims = (4,6,8,10)
data = np.random.rand(dims[0]*dims[1]*dims[2]*dims[3])
data = data.reshape(dims)      # 차원 3차원으로 변경
print("\n### element가 {}개이고 각 차원이 {}인 {}차원 텐서를 쪼개본다 ###".format(data.size, data.shape, data.ndim))
print("\n1. Original data \n{}".format(data.shape))
ft = torch.FloatTensor(data)
print(ft.shape)

tt_shape = to_tt_tensor(ft)
print(tt_shape)














#
# ######## CASE 1: Tensorly packages w/ 3차원데이터
# # TT Decomposition

# # Setting Hyper-parameters
# dims = (4,6,8)
# # dims = (2,2,2)
# # tt_rank = [1,4,3,1]     # len(rank) == tl.ndim(data)+1 쪼개고자하는 데이터가 3차원이면 rank는 4개가 필요/ 양 끝은 vector이므로  차원 1
# tt_rank = [1,3,4,1]     # len(rank) == tl.ndim(data)+1 쪼개고자하는 데이터가 3차원이면 rank는 4개가 필요/ 양 끝은 vector이므로  차원 1
#
# # 데이터 generation
# data = np.random.rand(dims[0]*dims[1]*dims[2])
# # data = np.arange(1,9)
# data = data.reshape(dims)      # 차원 3차원으로 변경
# print("\n### element가 {}개이고 각 차원이 {}인 {}차원 텐서를 쪼개본다 ###".format(data.size, data.shape, data.ndim))
# print("\n1. Original data \n{}".format(data.shape))
#
# tensor = tl.tensor(data)        # numpy array 상태인 데이터를  tensor 포멧으로 변경
# # factors = matrix_product_state(tensor, rank=tt_rank)       # matrix_product_state = tensor train decomposition 수행
# factors = tensor_train(tensor, rank=tt_rank)       # matrix_product_state = tensor train decomposition 수행
# # print(factors)
# print("\n2. After TT decomposition with ranks {}".format(tt_rank))
# print(factors)
# for i in range(0,len(factors)):
#     # print("Dim:{}\n{}".format(factors[i].shape, factors[i]))
#     print("Dim:{}\n".format(factors[i].shape))
#
#
#
#
# ######## CASE 2: Tensorly packages w/ 2차원데이터
# # Setting Hyper-parameters
# dims = (4,7)
# tt_rank = [1,2,1]     # len(rank) == tl.ndim(data)+1 쪼개고자하는 데이터가 2차원이면 rank는 3개가 필요/ 양 끝은 vector이므로  차원 1
#
#
# # 데이터 generation
# data = np.random.rand(dims[0]*dims[1])
# data = data.reshape(dims)      # 차원 3차원으로 변경
# print("\n### element가 {}개이고 각 차원이 {}인 {}차원 텐서를 쪼개본다 ###".format(data.size, data.shape, data.ndim))
# print("\n1. Original data \n{}".format(data.shape))
#
# # TT Decomposition
# tensor = tl.tensor(data)        # numpy array 상태인 데이터를  tensor 포멧으로 변경
# # factors = matrix_product_state(tensor, rank=tt_rank)       # matrix_product_state = tensor train decomposition 수행
# factors = tensor_train(tensor, rank=tt_rank)       # matrix_product_state = tensor train decomposition 수행
# # print(factors)
# print("\n2. After TT decomposition with ranks {}".format(tt_rank))
# print(factors)
# for i in range(0,len(factors)):
#     # print("Dim:{}\n{}".format(factors[i].shape, factors[i]))
#     print("Dim:{}\n".format(factors[i].shape))
#
#
#
#
#
#

# factors = parafac(tensor, rank=3, init='random', tol=10e-6, n_iter_max=25)
# print(factors[0], factors[1], factors[2])
# print(factors)
# print(factors[0].shape, factors[1].shape)

# tensor = tl.tensor([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#                         [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
#                         [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
#                         [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
#                         [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
#                         [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
#                         [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
#                         [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
#                         [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
#                         [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
#                         [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
#                         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
# print(tensor)
# # factors = parafac(tensor, rank=2)
# # print(factors)
#
# factors = matrix_product_state(tensor, rank=[1,2,1])
# print(factors)
# print(factors[0], factors[1])
# print(factors[0].shape, factors[1].shape)
