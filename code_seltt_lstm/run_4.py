import subprocess
import os



cmd_lstm = [
    ["python", "ttlstm_ele_test/test_4.py", "--mode", "0", "--n_layers", "2", "--n_front_layers", "2", "--lr", "0.0001", "--epochs", "100"],
    ["python", "ttlstm_ele_test/test_4.py", "--mode", "0", "--n_layers", "2", "--n_front_layers", "2", "--lr", "0.00001", "--epochs", "100"]
]

for i, cmd in enumerate(cmd_lstm):
    p = subprocess.Popen(cmd)
    p.wait()
print('lstm completed!')

cmd_ttlstm = [
    ["python", "ttlstm_ele_test/test_4.py", "--mode", "1", "--n_layers", "2", "--n_front_layers", "2", "--ncores", "2", "--ttrank", "2", "--lr", "0.0001", "--epochs", "100"],
    ["python", "ttlstm_ele_test/test_4.py", "--mode", "1", "--n_layers", "2", "--n_front_layers", "2", "--ncores", "2", "--ttrank", "2", "--lr", "0.00001", "--epochs", "100"]
]

for i, cmd in enumerate(cmd_ttlstm):
    p = subprocess.Popen(cmd)
    p.wait()
print('ttlstm completed!')

cmd_sellstm = [
    ["python", "ttlstm_ele_test/test_4.py", "--mode", "2", "--n_layers", "2", "--n_front_layers", "1", "--ncores", "2", "--ttrank", "2", "--lr", "0.0001", "--epochs", "100"],
    ["python", "ttlstm_ele_test/test_4.py", "--mode", "2", "--n_layers", "2", "--n_front_layers", "1", "--ncores", "2", "--ttrank", "2", "--lr", "0.00001", "--epochs", "100"]
]


for i, cmd in enumerate(cmd_sellstm):
    p = subprocess.Popen(cmd)
    p.wait()
print('sellstm completed!')




