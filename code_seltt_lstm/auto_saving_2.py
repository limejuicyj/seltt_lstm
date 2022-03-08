import pandas as pd
import numpy as np
import glob, re
import datetime


# 한 폴더내에 반드시 epoch이 동일한 파일만 있어야 에러 안남

##### Folder and File list #####
INPUT_FOLDER = r'/media/data1/sel_ttlstm/mnist_results/default/'
SAVING_FOLDER = r'./results/'
# INPUT_FOLDER = r'./results/epoch10/'
# INPUT_FOLDER = r'./results/epoch5/'
# INPUT_FILE = 'cifar10_'       # data name
INPUT_FILE = ''             # data name
# INPUT_FILE = 'lstm_'             # data name
all_files = glob.glob(INPUT_FOLDER + INPUT_FILE + '*.txt')

##### Printing Options #####
saving_flag = 1                     # Save to csv : 1, without saving : 0
saving_epoch = 3                    # print X times of epochs. if 1: print all / if 5: 5,10,15...


##### Word List #####
f1 = re.compile('File name')
f2 = re.compile('Namespace')
f3 = re.compile('Total')            # total num of parameters
f4 = re.compile('Average loss')     # 'Accuracy' is in the same line
f5 = re.compile('Runtime')
# f6 = re.compile('total = ')
# f7 = re.compile('param_last_linear')
lst = [f1,f2,f3,f4,f5]
df_final = []



##### Finding Words ######
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
    print("df:",df)
    print("df:",df[1])



    #### F1."File name" #####
    df_main = pd.DataFrame({"File name":[i]})

    ##### F2. "Namespace" #####
    file_name = i.split("/")
    file_name = file_name[len(file_name)-1].split("_")
    print("file name split :", file_name)
    f_dataset = file_name[0]
    print("f_dataset :", f_dataset)
    idx = 2
    if f_dataset == "lstm":
        idx = idx-1
    f_input_size = file_name[idx].split("in")[1]
    f_hidden_size = file_name[idx+5].split("h")[1]
    f_mode = file_name[idx+1]
    f_n_layer = (file_name[idx+2].split("(")[0]).split("n")[1]
    f_n_cores = file_name[idx + 3].split("ncores")[1]
    f_tt_ranks = file_name[idx + 4].split("r")[1]
    f_n_front_layer = ""
    if f_mode == "basic":
        f_n_cores = ""
        f_tt_ranks = ""
    elif f_mode == "sel":
        f_n_front_layer = (file_name[idx+2].split("(")[1]).split("+")[0]
    print("f_mode :{}".format(f_mode))
    print("f_n_layer :{} / f_n_front_layer :{} ".format(f_n_layer,f_n_front_layer))
    print("f_n_cores :{} / f_tt_ranks :{} ".format(f_n_cores,f_tt_ranks))

    lr_tmp = file_name[idx+6].split("lr")
    if lr_tmp[0] == "":
        f_lr = lr_tmp[1]
        f_epoch = file_name[idx + 7].split("ep")[1]
    else:
        f_lr = 0.01
        f_epoch = (file_name[idx + 6].split("ep")[1]).split(".")[0]
    print("f_lr :{} / f_epoch :{} ".format(f_lr,f_epoch))

    df_tmp = pd.DataFrame({"dataset": [f_dataset], "input_size": [f_input_size], "hidden_size": [f_hidden_size],
                            "epochs":[f_epoch], "mode":[f_mode], "n_layers":[f_n_layer], "n_front_layers":[f_n_front_layer],
                            "ncores":[f_n_cores], "ttrank":[f_tt_ranks], "lr":[f_lr]})

    df_main = pd.concat([df_main, df_tmp], axis=1)



    ##### F5. "Total" #####
    total_split = df[2].split(" ")
    total_param_num = total_split[len(total_split)-1].split("\n")[0]

    df_main["#parameters"] = [total_param_num]
    print("df_main" , df_main)
    num_cols = len(df_main.columns)
    print("num_cols",num_cols)


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
col_1 = ['dataset', 'input_size', 'mode', 'n_layers', 'n_front_layers', 'ncores', 'ttrank', 'epochs', 'lr', 'hidden_size', '#parameters']


col_tmp = df_final.columns[num_cols:].to_list()     #ALL loss and acc


col_2 = []                     # loss and acc for epoch 5,10,15,...
for i in range(0,len(col_tmp),1):
    if (i+3) % (3 * saving_epoch) == 0:
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

