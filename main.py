# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from src import helpers, implementations

data_path = './all/'
reload_data = False
path_dataset = data_path+r'train.csv'
data = []

if reload_data:
    data, y = helpers.load_data(data_path+r'train.csv')
#    test_data, te_col = load_data(data_path+r'test.csv')

#data = np.genfromtxt(path_dataset, delimiter=",", 
#                              usecols=range(2, 32), skip_header=1)
#
#y = np.genfromtxt(path_dataset, delimiter=',', usecols=1, 
#                      dtype=str, skip_header=1)
    
    


y, data, cols = helpers.load_data(path_dataset)


print(data.shape)
print(type(data), type(data[1]), '\n')

print(y.shape)
print(type(y), type(y[1]), '\n')

print(cols)

    













