import pandas as pd
import numpy as np
import glob, re
import datetime
import os


os.environ["CUDA_VISIBLE_DEVICES"] = '5'


# 한 폴더내에 반드시 epoch이 동일한 파일만 있어야 에러 안남

##### Folder and File list #####
# INPUT_FOLDER = r'/media/data1/sel_ttlstm/mnist_results/default/'
# INPUT_FOLDER = r'/media/data1/sel_ttlstm/mnist_results/lr/'
# INPUT_FOLDER = r'/media/data1/sel_ttlstm/mnist_results/front_layer/'
# INPUT_FOLDER = r'/media/data1/sel_ttlstm/mnist_results/titan_gpu/lr_rank_core/lr=0.001/'
# INPUT_FOLDER = r'/media/data1/sel_ttlstm/mnist_results/titan_gpu/rank_core/'
INPUT_FOLDER = r'/media/data1/sel_ttlstm/mnist_results/titan_gpu/rank_core/ep100/'
# INPUT_FOLDER = r'./results/'
# INPUT_FOLDER = r'./results/epoch10/'
# INPUT_FILE = 'cifar10_'       # data name
INPUT_FILE = ''             # data name
all_files = sorted(glob.glob(INPUT_FOLDER + INPUT_FILE + '*combined.txt'))
# all_files = sorted(glob.glob(INPUT_FOLDER + INPUT_FILE + '*.txt'))



##### Printing Options #####
saving_flag = 1                     # Save to csv : 1, without saving : 0
epoch_unit = 5                    # print X times of epochs. if 1: print all / if 5: 5,10,15...
# col_order_acc_first = 1             # 1 : all accs - all losses - all runtimes, 0 : loss-acc-runtime


##### Word List #####
f1 = re.compile('File name')
f2 = re.compile('Namespace')
f3 = re.compile('Total')            # total num of parameters
f4 = re.compile('Average loss')     # 'Accuracy' is in the same line
f5 = re.compile('Runtime')
# f6 = re.compile('total = ')
# f7 = re.compile('param_last_linear')
# lst = [f1,f2,f3,f4,f5]            # 'Runtime' is seperated
lst = [f1,f2,f3,f4]                 # 'Runtime' is in the same line with
df_final = []



##### Finding All Words ######
for i, file in enumerate(all_files):
    df = []
    # if i%2 == 1:
    with open(file, 'r') as f:
        # print("============================== ")
        for x, y in enumerate(f.readlines(),1):
            for index, field in enumerate(lst):
                ep_idx = 0
                m = field.findall(y)
                if m:
                    # if index == 0:              # 'index' is only for printing file name once
                    #     print("File name : %s" % (file))
                    #     print("---------------------------- ")
                    # print("WORD : {}  (line:{})".format(m,x))
                    # print('Full Line Text : %s' % y)
                    df.append(y)
    # print("---------------------------- ")
    # print("df:",df)
    # print("df:",df[1])

    runtime_list = [s for s in df if "Runtime" in s]        # list for runtime
    accuracy_list = [s for s in df if "Accuracy" in s]      # list for accuracy and loss
    # print("runtime_list:", runtime_list, len(runtime_list))





    #### F1."File name" #####
    df_main = pd.DataFrame({"File name":[file]}
                           )

    ##### F2. "Namespace" #####
    file_name = file.split("/")
    file_name = file_name[len(file_name)-1].split("_")
    print("========================================")
    print("file name split :", file_name)
    f_dataset = file_name[0]
    idx = 2
    if f_dataset == "lstm":
        idx = idx-1
        f_dataset = 'mnist'
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
    lr_tmp = file_name[idx+6].split("lr")
    if lr_tmp[0] == "":
        f_lr = lr_tmp[1]
        f_epoch = file_name[idx + 7].split("ep")[1]
    else:
        f_lr = 0.01
        f_epoch = (file_name[idx + 6].split("ep")[1]).split(".")[0]
    print("f_dataset: {} / f_mode: {} / f_lr: {}".format(f_dataset, f_mode, f_lr))
    print("f_n_layer (front): {} ({}) / f_n_cores: {} / f_tt_ranks: {} ".format(f_n_layer,f_n_front_layer, f_n_cores,f_tt_ranks))
    print("f_input_size: {} / f_hidden_size: {} / f_epoch: {}".format(f_input_size,f_hidden_size,f_epoch))

    df_tmp = pd.DataFrame({"dataset": [f_dataset], "input_size": [f_input_size], "hidden_size": [f_hidden_size],
                            "epochs":[f_epoch], "mode":[f_mode], "n_layers":[f_n_layer], "n_front_layers":[f_n_front_layer],
                            "ncores":[f_n_cores], "ttrank":[f_tt_ranks], "lr":[f_lr]})

    df_main = pd.concat([df_main, df_tmp], axis=1)
    print("df_main.columns : ",df_main.columns)


    ##### F5. "Total" #####
    total_split = df[2].split(" ")
    total_param_num = total_split[len(total_split)-1].split("\n")[0]

    df_main["#parameters"] = [total_param_num]
    print("\ndf_main" , df_main)
    num_cols = len(df_main.columns)
    # print("df_main>> # of cols: {} ({})".format(num_cols,df_main.columns.values))



    ##### F6. "Average loss (Accuracy) and Runtime" #####
    # for k in range(3,len(runtime_list)):
    for k in range(0,len(runtime_list)):
        if (k+1) % epoch_unit == 0:
            df_main["accuracy_" + str('{0:03d}'.format(k+1))] = (accuracy_list[k].split("(")[1]).split("%")[0]
            df_main["avg_loss_" + str('{0:03d}'.format(k+1))] = (accuracy_list[k].split("Average loss: ")[1]).split(", Accuracy")[0]
            df_main["runtime_" + str('{0:03d}'.format(k+1))] = (runtime_list[k].split("Runtime: ")[1]).split(" sec")[0]

    print(df_main)
    df_final = pd.concat([pd.DataFrame(df_final),df_main])



##### #####
print("FINAL_draft : \n", df_final)       # including all information



##### Save to CSV #####
basename = "results_" + INPUT_FILE + str(len(all_files)) + "files"
suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
filename = "_".join([basename, suffix])



if saving_flag == 1:
    saving_file_name = INPUT_FOLDER + filename +'.csv'
    df_final.to_csv(saving_file_name, index = False)
    print("SAVED as : ", saving_file_name)

