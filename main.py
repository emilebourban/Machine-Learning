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

#Logistic regression
if log_reg:
    data = input_data.copy()
         #Removes the col with to many error values
    n_del = 0
    range_ = range(data.shape[1])
    for i in range_:    #size changes, we want every colums
        if (percent_nan[i] ) >= 70:
            data = np.delete(data, (i-n_del),  axis=1)
            data_pr = np.delete(data_pr, (i-n_del),  axis=1)
            n_del +=1
    data = np.delete(data, 22,  axis=1)
    imp.outliers_to_mean(data)
    #imp.remove_outliers(data)
    data = imp.standardize(data)
    gammas = np.logspace(-5, -3, 10)
    #gammas = np.array([0.00000001, 0.00001])
    max_iter = 1000
     
    accuracy_poly_log_reg = np.zeros((len(degree), gammas.shape[0]))
    for d in range(len(degree)): 
        data_poly = imp.build_poly(data, degree[d])
        y_tr, y_te, data_poly_tr, data_poly_te = help1.split_data(yb, data_poly, k_fold)
        initial_w = np.ones(data_poly_tr.shape[2])
        ws_poly_log_reg = np.zeros((k_fold, data_poly_tr.shape[2])); losses_poly_log_reg = [];
        for g in range(gammas.shape[0]):
            
            for k in range(k_fold):
                w_temp, loss_temp = imp.logistic_regression(y_tr[k], data_poly_tr[k], initial_w, max_iter, gammas[g])
                ws_poly_log_reg[k] = np.array(w_temp)
                losses_poly_log_reg.append(loss_temp)
            w_poly_log_reg_mean = np.array(ws_poly_log_reg.mean(axis=0))
            accuracy = []
            for k in range(k_fold):
                accuracy.append(imp.calculate_accuracy(data_poly_te[k], y_te[k], w_poly_log_reg_mean))
            accuracy_poly_log_reg[d, g] = np.array(accuracy).mean()
            print('Accuracy for {} degree poly and gamma = {} is {}'.format(\
                  degree[d], gammas[g], np.array(accuracy).mean()))
            
    temp_ind = np.where(accuracy_poly_log_reg == accuracy_poly_log_reg.max())
    best_acc_ind = (temp_ind[0][0],temp_ind[1][0])
    best_acc = accuracy_poly_log_reg[best_acc_ind]
    best_gamma = gammas[best_acc_ind[1]]
    best_deg = degree[best_acc_ind[0]]
    print('the best accuracy is {} with degree {} and gamma {}'.format(\
            best_acc, best_deg, best_gamma))
    
    #generate figure of accuracy depending on gamma values for different polynomial degrees
    fig6 = plt.figure(6)
    ax6 = fig6.add_subplot(111)
    ax6.set_xscale('log')
    for d in range(len(degree)):
        plt.plot(gammas, accuracy_poly_log_reg[d, :], label='Accuracy w/ poly deg: '+str(degree[d])) 
    plt.title('Accuracy for logistic regression')
    plt.xlabel('gamma'); plt.ylabel('Accuracy');
    plt.legend(loc=0)
    plt.show()

#Ridge regression with poly for different lambdas
if poly_ridge:
    
    # Removes the col with to many error values
#    n_del = 0
#    range_ = range(data.shape[1])
#    for i in range_:    #size changes, we want every colums
#        if (percent_nan[i] ) >= 70:
#            data = np.delete(data, (i-n_del),  axis=1)
#            data_pr = np.delete(data_pr, (i-n_del),  axis=1)
#            n_del +=1
    #data = np.delete(data, 22,  axis=1)
    
    lambdas = np.logspace(-8, -1, 20)
    accuracy_poly_ridge = np.zeros((len(degree), lambdas.shape[0]))
    for d in range(len(degree)):
        data_poly = imp.build_poly(data, degree[d])
        y_tr, y_te, data_poly_tr, data_poly_te = help1.split_data(yb, data_poly, k_fold)
        
        ws_poly_ridge = np.zeros((k_fold, data_poly_tr.shape[2])); losses_poly_ridge = []; 
        for l in range(lambdas.shape[0]):
            for k in range(k_fold):
                w_temp, loss_temp = imp.ridge_regression(y_tr[k], data_poly_tr[k], lambdas[l])
                ws_poly_ridge[k] = np.array(w_temp)
                losses_poly_ridge.append(loss_temp)
            w_poly_ridge_mean = np.array(ws_poly_ridge.mean(axis=0))
            accuracy = []
            for k in range(k_fold):
                accuracy.append(imp.calculate_accuracy(data_poly_te[k], y_te[k], w_poly_ridge_mean))
                
            accuracy_poly_ridge[d ,l] = np.array(accuracy).mean()
            print('Accuracy for {} degree poly and lambda = {} is {}'.format(\
                  degree[d], lambdas[l], np.array(accuracy).mean()))
    temp_ind = np.where(accuracy_poly_ridge == accuracy_poly_ridge.max())
    best_acc_ind = (temp_ind[0][0],temp_ind[1][0])
    best_acc = accuracy_poly_ridge[best_acc_ind]
    best_lambda = lambdas[best_acc_ind[1]]
    best_deg = degree[best_acc_ind[0]]
    print('the best accuracy is {} with degree {} and lambda {}'.format(\
            best_acc, best_deg, best_lambda))
#figure generation of accuracy depending on lambda values with different polynomial degrees    
    fig5 = plt.figure(3)
    ax5 = fig5.add_subplot(111)
    ax5.set_xscale('log')
    for d in range(len(degree)):
        plt.plot(lambdas, accuracy_poly_ridge[d, :], label='Accuracy w/ poly deg: '+str(degree[d])) 
    plt.title('Accuracy for polynomial reg')
    plt.xlabel('lambda'); plt.ylabel('Accuracy');
    plt.legend(loc=0)
    plt.show()

# Sets error limit [%] and creates data copy with selected column
err_lim = [100]; 
degree = [5];
accuracy_reg = []; accuracy_off = [];  
accuracy_poly = np.zeros((len(err_lim), len(degree)));
# copies and cleans the data so we don't have to import each time
data_pr = imp.standardize(input_data_pr.copy())
data = imp.remove_outliers(imp.standardize(input_data.copy()));

for e in range(len(err_lim)):
    print("Computing w/ err limit =", err_lim[e])
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
    
        for d in range(len(degree)):
            data_poly = imp.build_poly(data, degree[d])
            y_tr, y_te, data_poly_tr, data_poly_te = help1.split_data(yb, data_poly, k_fold)
            
            ws_poly = []; losses_poly = [];
            for i in range(k_fold):
                w, loss = imp.least_squares(y_tr[i], data_poly_tr[i])
                ws_poly.append(w); losses_poly.append(loss)                    
            
            w_poly_mean = np.array(ws_poly).mean(axis=0)   
            accuracy_poly[e, d] = imp.calculate_accuracy(data_poly_te[i], y_te[i], w_poly_mean)       
            # Saves the weight for what we knpw are the best parameters
            if (degree[d] == 5 and err_lim[e] == 100):
                w_best = w_poly_mean
                print("{} % accurate".format(accuracy_poly[e, d]*100))
             

# PLots the accuracy for both models so that we can find the best error threshold
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
plt.plot(err_lim, accuracy_reg, label='Accuracy w/out offset')
plt.plot(err_lim, accuracy_off, label='Accuracy w/ offset')
plt.title('Accuracy in function of the error limit on column')
plt.xlabel('% Error allowed'); plt.ylabel('Accuracy');   
plt.legend(loc=0)
for i in range(len(err_lim)):
    ax2.annotate(str(accuracy_off[i]),xy=(err_lim[i]-5, accuracy_off[i]+0.002))
plt.show()

fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)
for d in range(len(degree)):
    plt.plot(err_lim, accuracy_poly[:, d], label='Accuracy w/ poly deg: '+str(degree[d])) 
plt.title('Accuracy for polynomial reg')
plt.xlabel('% Error allowed'); plt.ylabel('Accuracy');
plt.legend(loc=0)
for i in range(len(err_lim)):
    ax3.annotate(str(accuracy_poly[i, -1]),xy=(err_lim[i]-5, accuracy_poly[i, -1]+0.002))
plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
