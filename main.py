# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from src import implementations as imp
from src import proj1_helpers as help1

data_path = './all/'
reload_data = False
all_cols = True
under_40 = True
under_20 = True
under_10 = True

if reload_data:
    yb, input_data, ids = help1.load_csv_data(data_path+r'train.csv')
    yb_te, input_data_te, ids_te = help1.load_csv_data(data_path+r'test.csv')


percent_nan = []
for i in range(input_data.shape[1]):
    c = 0
    for j in range(input_data.shape[0]):    
        if (input_data[j][i] == 0):
            c +=1
    percent_nan = np.r_[percent_nan, (c * 100)/input_data.shape[0]]       
    print(percent_nan[i], "% wrong values in col:", i)
    
    
if all_cols:
    w, loss = imp.least_squares(yb, input_data)
    print("Taking all colomns", '\n w* = \n', w, '\n loss =', loss)
    print("loss test =", imp.compute_loss(yb_te, input_data_te, w))
        
if under_40:
    data = np.delete(input_data, [4,5,6,12,26,27,28,29], axis=1)
    data_te = np.delete(input_data_te, [4,5,6,12,26,27,28,29], axis=1)
    w2, loss2 = imp.least_squares(yb, data)
    print("<40% wrong values", '\n w* = \n', w2, '\n loss =', loss2)
    print("loss test =", imp.compute_loss(yb_te, data_te, w2))
    
    
if under_20:
    data = np.delete(input_data, [4,5,6,12,22,23,24,25,26,27,28,29], axis=1)
    data_te = np.delete(input_data_te, [4,5,6,12,22,23,24,25,26,27,28,29], axis=1)
    w3, loss3 = imp.least_squares(yb, data)
    print("<20% wrong values", '\n w* = \n', w3, '\n loss =', loss3)
    print("loss test =", imp.compute_loss(yb_te, data_te, w3))

if under_10:
    data = np.delete(input_data, [0,4,5,6,12,22,23,24,25,26,27,28,29], axis=1)
    data_te = np.delete(input_data_te, [0,4,5,6,12,22,23,24,25,26,27,28,29], axis=1)
    w3, loss3 = imp.least_squares(yb, data)
    print("<10% wrong values", '\n w* = \n', w3, '\n loss =', loss3)
    print("loss test =", imp.compute_loss(yb_te, data_te, w3))
    
if under_10:
    data = np.delete(input_data, [0,4,5,6,12,22,23,24,25,26,27,28,29], axis=1)
    data_te = np.delete(input_data_te, [0,4,5,6,12,22,23,24,25,26,27,28,29], axis=1)
    np.append(data, np.ones(data[0].shape[0]))
    w3, loss3 = imp.least_squares(yb, data)
    print("<10% wrong values  w/ offset", '\n w* = \n', w3, '\n loss =', loss3)
    print("loss test =", imp.compute_loss(yb_te, data_te, w3))





