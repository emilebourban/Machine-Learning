# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from src import implementations as imp
from src import proj1_helpers as help1

data_path = './all/'
# Set to false after first run !
reload_data = False
offset= True
poly = True

# Only reload once, takes a lot of time
if reload_data:
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

# Sets error limit [%] and creates data copy with selected column
err_lim = [100, 75, 65, 35, 10, 1e-6]; 
accuracy_reg = []; accuracy_off = [];
data = input_data.copy(); data_pr = input_data_pr.copy()

for e in range(len(err_lim)):
    
    n_del = 0
    # Removes the col with to many error values
    range_ = range(data.shape[1])
    for i in range_:    #size changes, we want every colums
        if (percent_nan[i] + percent_zeros[i]) >= err_lim[e]:
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
            
    # Takes the mean w as the best approximation (test w/ median)
    w_ls = np.array(ws).mean(axis=0)
    accuracy_reg.append(imp.calculate_accuracy(data_te[0], y_te[0], w_ls))
    
    # Same as before but with off set in the data
    if offset:
        data_off = np.c_[np.ones(data.shape[0]), data]    
        y_tr, y_te, data_tr_off, data_te_off = help1.split_data(yb, data_off, k_fold)

        ws_off = []; losses_off = [];
        for i in range(k_fold):
                # Loss values for all the data and cleaned data
                w, loss = imp.least_squares(y_tr[i], data_tr_off[i])
                ws_off.append(w); losses_off.append(loss)
    
    # Finds the accuracy        
    w_ls_off = np.array(ws_off).mean(axis=0)
    accuracy_off.append(imp.calculate_accuracy(data_te_off[0], y_te[0], w_ls_off))
    
    if e == 0:
        w_temp = w_ls_off
        
    if poly:
    
#        data = input_data.copy();
#        y_tr, y_te, data_tr, data_te = help1.split_data(yb, data, k_fold)
        degree = [1]; ws_poly = []
        for d in range(len(degree)):
            print("degree:", degree[d])
            for i in range(k_fold):
                poly_data_tr = imp.build_poly(data_tr[i], degree[d])
                poly_data_te = imp.build_poly(data_te[i], degree[d])
                print(poly_data_tr.shape, poly_data_te.shape)
                w_poly, loss_poly = imp.least_squares(y_tr[i], poly_data_tr)
                ws_poly.append(w_poly)            
                w_poly_mean = np.array(ws_poly).mean(axis=0)   
            
                acc_poly = imp.calculate_accuracy(poly_data_te[i], y_te[i], w_poly_mean)
                print("loss=", loss_poly,"accuracy:", acc_poly)



# PLots the accuracy for both models so that we can find the best error threshold
fig = plt.figure(2)
ax = fig.add_subplot(111)
ax1 = plt.plot(err_lim, accuracy_reg, label='Accuracy w/out offset')
ax2 = plt.plot(err_lim, accuracy_off, label='Accuracy w/ offset') 
plt.title('Accuracy in function of the error limit on column')
plt.xlabel('% Error allowed'); plt.ylabel('Accuracy');   
plt.legend(loc=0)
for i in range(len(err_lim)):
    ax.annotate(str(accuracy_reg[i]),xy=(err_lim[i]-5, accuracy_reg[i]-0.002))
    ax.annotate(str(accuracy_off[i]),xy=(err_lim[i]-5, accuracy_off[i]+0.002))
plt.show()





