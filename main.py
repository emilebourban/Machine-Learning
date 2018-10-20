# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from src import implementations as imp
from src import proj1_helpers as help1

data_path = './all/'
# Set to false after first run !
reload_data = False
offset= True

# Only reload once, takes a lot of time
if reload_data:
    yb, input_data, ids = help1.load_csv_data(data_path+r'train.csv')
    yb_pr, input_data_pr, ids_pr = help1.load_csv_data(data_path+r'test.csv')

# Computes the amount of wrong values (-999 and 0) in the input data
percent_nan = np.count_nonzero(input_data == -999.0, axis=0)/(input_data.shape[0]*0.01)  
percent_zeros = np.count_nonzero(input_data == 0.0, axis=0)/(input_data.shape[0]*0.01)

data = input_data.copy(); data_pr = input_data_pr.copy()

# Prints the total percent error, for evry column
for i in range(len(percent_nan)):
    print(percent_nan[i] + percent_zeros[i], "% wrong values in col:", i)

# Sets error limit [%] and creates data copy with selected column
err_lim = 35; n_del = 0; 

# Removes the col with to many error values
for i in range(input_data.shape[1]):
    if (percent_nan[i] + percent_zeros[i]) > err_lim:
        data = np.delete(data, (i-n_del),  axis=1)
        data_pr = np.delete(data_pr, (i-n_del),  axis=1)
        n_del +=1

#Splits the data into k_fold 
k_fold = 5
y_tr, y_te, data_tr, data_te = help1.split_data(yb, data, k_fold)


ws = []; losses = [];
for i in range(k_fold):
        # Loss values for all the data and cleaned data
        w, loss = imp.least_squares(y_tr[i], data_tr[i])
        ws.append(w); losses.append(loss) 


w_ls = np.array(ws).mean(axis=0)
acc = imp.calculate_accuracy(data_te[0], y_te[0], w_ls)
print("accuracy:", acc)

if offset:
    data_off = np.c_[np.ones(data.shape[0]), data]
    
    k_fold = 5
    _, _, data_tr_off, data_te_off = help1.split_data(yb, data_off, k_fold)


    ws_off = []; losses_off = []; accuracy = 0;
    for i in range(k_fold):
            # Loss values for all the data and cleaned data
            w, loss = imp.least_squares(y_tr[i], data_tr_off[i])
            ws_off.append(w); losses_off.append(loss)
           
    w_ls_off = np.array(ws_off).mean(axis=0)
    acc_off = imp.calculate_accuracy(data_te_off[0], y_te[0], w_ls_off)  
    print("accuracy w/ offset (mean w):",acc_off)








