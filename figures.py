# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from src import implementations
from src import proj1_helpers

data_path = './all/'
# Set to false after first run !
reload_data = False
offset= True
poly = True

err_lim = [100, 75, 65, 35, 10, 1e-6]
degree = [1, 2, 4, 5, 6]; degree_illfit = [1, 3, 6, 7, 8, 9];


# Only reload once, takes a lot of time
if reload_data:<
    yb, input_data, ids = help1.load_csv_data(data_path+r'train.csv')
    yb_pr, input_data_pr, ids_pr = help1.load_csv_data(data_path+r'test.csv')

# Computes the amount of wrong values (-999 and 0) in the input data
percent_nan = np.count_nonzero(input_data == -999.0, axis=0)/(input_data.shape[0]*0.01)  
percent_zeros = np.count_nonzero(input_data == 0.0, axis=0)/(input_data.shape[0]*0.01)

# Dispalys the total percent error, for evry column
plt.figure(1)
xplot = np.linspace(0, len(percent_nan), len(percent_nan));
plt.bar(xplot, percent_nan, label='Nan Values');
plt.bar(xplot, percent_zeros, label='Zero values')
plt.title('Error percentage for each colums')
plt.xlabel('Columns'); plt.ylabel('% Values');
plt.legend(loc=1)
plt.show()



