# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from src import implementations as imp
from src import proj1_helpers as help1

data_path = './all/'
# Set to false after first run !
reload_data = True
all_cols = True
under_40 = True
under_20 = True
under_10 = True

# Only reload once, takes a lot of time
if reload_data:
    yb, input_data, ids = help1.load_csv_data(data_path+r'train.csv')
    yb_te, input_data_te, ids_te = help1.load_csv_data(data_path+r'test.csv')

# Computes the amount of wrong values (-999 and 0) in the input data
percent_nan = np.count_nonzero(input_data == -999, axis=0)/(input_data.shape[0]*0.01)  
percent_zeros = np.count_nonzero(input_data == 0, axis=0)/(input_data.shape[0]*0.01)
    
# Prints the total percent error, for evry column
for i in range(len(percent_nan)):
    print(percent_nan[i]+ percent_zeros[i], "% wrong values in col:", i)

# Sets error limit [%] and creates data copy with selected column
err_lim = 70; n_del = 0
data = input_data.copy(); data_te = input_data_te.copy()

for i in range(input_data.shape[1]):
    if (percent_nan[i] + percent_zeros[i]) > err_lim:
        data = np.delete(data, [i-n_del],  axis=1)
        data_te = np.delete(data_te, [i-n_del],  axis=1)
        n_del +=1
        
        
        
print(data.shape)

# Loss values for all the data and cleaned data
w, loss = imp.least_squares(yb, input_data)
print("Taking all columns", '\n w* = \n', w, '\n loss =', loss)
print("loss test =", imp.compute_loss(yb_te, input_data_te, w), end='\n\n')

w_clean, loss_clean = imp.least_squares(yb, data)
print("Taking column w/ error rate <", err_lim, '\n w* = \n', w_clean, '\n loss =', loss_clean)
print("loss test =", imp.compute_loss(yb_te, data_te, w_clean), end='\n\n')

offset = True

if offset:
    print('\n', "W/ offset", '\n\n')
    data = np.c_[np.ones(data.shape[0]), data]
    data_te = np.c_[np.ones(data_te.shape[0]), data_te]
    w_off, loss_off = imp.least_squares(yb, data)
    print("Taking column w/ error rate <", err_lim, '\n w* = \n', w_off, '\n loss =', loss_off)
    print("loss test =", imp.compute_loss(yb_te, data_te, w_off))
        




