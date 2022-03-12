import pandas as pd
import numpy as np
import glob, re
import datetime
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '6'

# 한 폴더내에 반드시 epoch이 동일한 파일만 있어야 에러 안남

##### Folder and File list #####
# INPUT_FOLDER = r'/media/data1/sel_ttlstm/mnist_results/titan_gpu/lr_rank_core/lr=0.001/'
# INPUT_FOLDER = r'/media/data1/sel_ttlstm/mnist_results/titan_gpu/rank_core/'

INPUT_FOLDER = r'/media/data1/sel_ttlstm/mnist_results/lr_rank_core/lr=0.001/'

INPUT_FILE = ''             # data name

all_files = sorted(glob.glob(INPUT_FOLDER + INPUT_FILE + '*.txt'))
all_files_combined = sorted(glob.glob(INPUT_FOLDER + INPUT_FILE + '*combined.txt'))
# all_files_origin = all_files
for rm_file in all_files_combined:
    all_files.remove(rm_file)

all_files_pair=[]
for i in range(0,len(all_files)-1,2):
    all_files_pair.append([all_files[i],all_files[i+1]])
print(all_files_pair)

for filename in all_files_pair:

    with open(filename[0][:-4]+'_combined.txt', 'w') as outfile:
        with open(filename[0]) as file:
            print("HE", filename[0])
            for line in file:
                outfile.write(line)
        with open(filename[1]) as file:
            print("HE", filename[1])
            for line in file:
                outfile.write(line)