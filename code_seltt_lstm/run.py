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
# p = subprocess.run(["python","test_2.py","--tt","7"])
# p = subprocess.Popen(['python','pmnist_test_2.py --n_layers 7'])
# subprocess.Popen(r"/home/youjin/research/tensor_new/code_ref_4_tensorized_rnn/experiments/exp_mnist/pmnist_test_2.py --n_layers=7", shell=True)

# import pmnist_test_2
# pmnist_test_2.__main__()




cmd_lstm = [
    ["python", "test_2.py", "--mode", "0", "--n_layers", "2", "--n_front_layers", "2", "--epochs", "50"],
    ["python", "test_2.py", "--mode", "0", "--n_layers", "3", "--n_front_layers", "3", "--epochs", "50"],
    ["python", "test_2.py", "--mode", "0", "--n_layers", "4", "--n_front_layers", "4", "--epochs", "50"],
    ["python", "test_2.py", "--mode", "0", "--n_layers", "5", "--n_front_layers", "5", "--epochs", "50"],
    ["python", "test_2.py", "--mode", "0", "--n_layers", "6", "--n_front_layers", "6", "--epochs", "50"],
    ["python", "test_2.py", "--mode", "0", "--n_layers", "7", "--n_front_layers", "7", "--epochs", "50"]
]

for i, cmd in enumerate(cmd_lstm):
    p = subprocess.Popen(cmd)
    p.wait()
print('lstm completed!')

cmd_ttlstm = [
    ["python", "test_2.py", "--mode", "1", "--n_layers", "2", "--n_front_layers", "2", "--ncores", "2", "--ttrank", "2", "--epochs", "50"],
    ["python", "test_2.py", "--mode", "1", "--n_layers", "3", "--n_front_layers", "3", "--ncores", "2", "--ttrank", "2", "--epochs", "50"],
    ["python", "test_2.py", "--mode", "1", "--n_layers", "4", "--n_front_layers", "4", "--ncores", "2", "--ttrank", "2", "--epochs", "50"],
    ["python", "test_2.py", "--mode", "1", "--n_layers", "5", "--n_front_layers", "5", "--ncores", "2", "--ttrank", "2", "--epochs", "50"],
    ["python", "test_2.py", "--mode", "1", "--n_layers", "6", "--n_front_layers", "6", "--ncores", "2", "--ttrank", "2", "--epochs", "50"],
    ["python", "test_2.py", "--mode", "1", "--n_layers", "7", "--n_front_layers", "7", "--ncores", "2", "--ttrank", "2", "--epochs", "50"]
]

for i, cmd in enumerate(cmd_ttlstm):
    p = subprocess.Popen(cmd)
    p.wait()
print('ttlstm completed!')

cmd_sellstm = [
    ["python", "test_2.py", "--mode", "2", "--n_layers", "2", "--n_front_layers", "1", "--ncores", "2", "--ttrank", "2", "--epochs", "50"],
    ["python", "test_2.py", "--mode", "2", "--n_layers", "3", "--n_front_layers", "1", "--ncores", "2", "--ttrank", "2", "--epochs", "50"],
    ["python", "test_2.py", "--mode", "2", "--n_layers", "4", "--n_front_layers", "1", "--ncores", "2", "--ttrank", "2", "--epochs", "50"],
    ["python", "test_2.py", "--mode", "2", "--n_layers", "5", "--n_front_layers", "1", "--ncores", "2", "--ttrank", "2", "--epochs", "50"],
    ["python", "test_2.py", "--mode", "2", "--n_layers", "6", "--n_front_layers", "1", "--ncores", "2", "--ttrank", "2", "--epochs", "50"],
    ["python", "test_2.py", "--mode", "2", "--n_layers", "7", "--n_front_layers", "1", "--ncores", "2", "--ttrank", "2", "--epochs", "50"]
]


for i, cmd in enumerate(cmd_sellstm):
    p = subprocess.Popen(cmd)
    p.wait()
print('sellstm completed!')


