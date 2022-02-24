# -*- coding: utf-8 -*-
# from subprocess import call
#
# call(["python","pmnist_test_2.py"])

# import os
# command = 'python myOtherScript.py ' + sys.argv[1] + ' ' + sys.argv[2]
# os.system(command)

# os.system("python pmnist_test_2.py")
#!/usr/bin/python
# exec(open("pmnist_test_2.py --n_layers 7").read())
# os.system("pmnist_test_2.py")


import subprocess
# print(__file__)
# print(os.path.realpath(__file__))
# print(os.path.abspath(__file__))
# print(os.getcwd())
# print(os.path.dirname(os.path.realpath(__file__)) )
p = subprocess.run(["python","test_2.py","--tt","7"])
# p = subprocess.Popen(['python','pmnist_test_2.py --n_layers 7'])
# subprocess.Popen(r"/home/youjin/research/tensor_new/code_ref_4_tensorized_rnn/experiments/exp_mnist/pmnist_test_2.py --n_layers=7", shell=True)

# import pmnist_test_2
# pmnist_test_2.__main__()