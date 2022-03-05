import pandas as pd
import numpy as np
import glob, re
import datetime


# 한 폴더내에 반드시 epoch이 동일한 파일만 있어야 에러 안남

##### Folder and File list #####
INPUT_FOLDER = r'./results/'
# INPUT_FOLDER = r'./results/epoch10/'
# INPUT_FOLDER = r'./results/epoch5/'
INPUT_FILE = 'cifar10_'       # data name
# INPUT_FILE = ''             # data name
all_files = glob.glob(INPUT_FOLDER + INPUT_FILE + '*.txt')

##### Printing Options #####
saving_flag = 1                     # Save to csv : 1, without saving : 0
saving_epoch = 1                    # print X times of epochs. if 1: print all / if 5: 5,10,15...


##### Word List #####
f1 = re.compile('File name')
f2 = re.compile('Namespace')
f3 = re.compile('Total')            # total num of parameters
f4 = re.compile('Average loss')     # 'Accuracy' is in the same line
f5 = re.compile('Runtime')
# f6 = re.compile('total = ')
# f7 = re.compile('param_last_linear')
lst = [f1,f2,f3,f4,f5]



##### Finding Words ######
df_final = []
for i in all_files:
    df = []
    with open(i, 'r') as f:
        # print("============================== ")
        for x, y in enumerate(f.readlines(),1):
            for index, field in enumerate(lst):
                ep_idx = 0
                m = field.findall(y)
                if m:
                    # if index == 0:              # 'index' is only for printing file name once
                    #     print("File name : %s" % (i))
                    #     print("---------------------------- ")
                    # print("WORD : {}  (line:{})".format(m,x))
                    # print('Full Line Text : %s' % y)
                    df.append(y)
    # print("---------------------------- ")
    # print("df:",df)



    #### F1."File name" #####
    df_main = pd.DataFrame({"File name":[df[0]]})

    ##### F2. "Namespace" #####
    df[1] = df[1].replace(" ", "")
    name_split = df[1].split(",")
    name_split[0] = name_split[0].split("(")[1]
    name_split[len(name_split)-1] = name_split[len(name_split)-1].split(")")[0]
    # print("\nAfter split:", name_split)

    cols=[]
    vals=[]
    for i in range(0,len(name_split)):
        name_split[i] = name_split[i].split("=")
        cols.append(name_split[i][0])
        vals.append(name_split[i][1])
        # print("cols: ",cols)
        # print("vals: ",vals)
    df_tmp = pd.DataFrame(vals).transpose()
    df_tmp.columns = cols
    # print("df_tmp : ", df_tmp)

    df_main = pd.concat([df_main, df_tmp], axis=1)


    ##### F5. "Total" #####
    total_split = df[2].split(" ")
    total_param_num = total_split[len(total_split)-1].split("\n")[0]

    df_main["#parameters"] = [total_param_num]
    # print(df_main)



    ##### F6. "Average loss (Accuracy) and Runtime" #####
    for i in range(3,len(df)):
        if (i%2 == 1):              # Average loss (Accuracy)
            loss_split = df[i].split(" ")
            # print(i, loss_split)
            # print("Average loss : ",loss_split[4])
            # print("Accuracy : ",int(loss_split[6].split("/")[0])/100)
            df_main["avg_loss_" + str(int(i / 2))] = loss_split[4].split(",")[0]
            df_main["accuracy_"+ str(int(i / 2))] = int(loss_split[6].split("/")[0])/100
        else:                       # Runtime
            df[i] = df[i].split(": ")[1]
            run_split = df[i].split(" sec")[0]
            df_main["runtime_"+ str(int(i / 2) - 1 )] = run_split


    df_final = pd.concat([pd.DataFrame(df_final),df_main])



##### #####
print("FINAL_draft : \n", df_final)       # including all information




##### Col Extraction #####
# col_1 = ['mode', 'File name', 'n_layers', 'n_front_layers', 'ncores', 'ttrank', 'epochs', 'lr', 'batch_size', 'hidden_size']
col_1 = ['mode', 'n_layers', 'n_front_layers', 'ncores', 'ttrank', 'epochs', 'lr', 'batch_size', 'hidden_size']


col_tmp = df_final.columns[21:].to_list()     # #parameters, and ALL loss and acc


col_2 = ['#parameters']                     # #parameters, and loss and acc for epoch 5,10,15,...
for i in range(0,len(col_tmp)):
    if (i + 2 ) % (3 * saving_epoch) == 0:
        col_2.append(col_tmp[i])
        col_2.append(col_tmp[i+1])
        col_2.append(col_tmp[i+2])


final_cols = col_1 + col_2
df_final = df_final[final_cols]
print("FINAL : \n", df_final)


##### Save to CSV #####
basename = INPUT_FILE + "results"
suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
filename = "_".join([basename, suffix])



if saving_flag == 1:
    df_final.to_csv(INPUT_FOLDER + filename +'.csv', index = False)
    print("SAVED as : ", INPUT_FOLDER + filename, ".csv")

