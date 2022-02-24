# tf2.0
# from __future__ import absolute_import, division, print_function, unicode_literals

# import tensorflow as tf

import torch
# from mnist_classifier import MNIST_Classifier


print(torch.cuda.is_available())

if torch.cuda.is_available():
    # set default cuda device.
    device = torch.device('cuda:{}'.format(0))
    torch.cuda.set_device(device)
    print("YJ:cuda is working well")
    # warn if not using cuda and gpu is available.
    if not True:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
else:
    device = torch.device('cpu')

# model = MNIST_Classifier(28, n_classes, 256, 1, device,
#                          tt=args.tt, gru=args.gru, n_cores=args.ncores,
#                          tt_rank=args.ttrank, naive_tt=args.naive_tt,
#                          log_grads=args.log_grads, extra_core=args.extra_core)
# model.cuda()




#
# tf.debugging.set_log_device_placement(True)
#
# # 텐서 생성
# a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# c = tf.matmul(a, b)
#
# print("hello")
# print(c)