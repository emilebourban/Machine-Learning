# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from src import implementations as imp
from src import proj1_helpers as help1

DATA_PATH = './all/'
# Set to false after first run !
RELOAD_DATA = True
OFFSET = True
POLY = True
LOG_REGRESS = True
POLY_RIDGE = True

#%%
# Only reload once, takes a lot of time
if RELOAD_DATA:
    Y_IN, INPUT_DATA, IDS = help1.load_csv_data(DATA_PATH+r'train.csv')
    Y_IN_PR, INPUT_DATA_PR, IDS_PR = help1.load_csv_data(DATA_PATH+r'test.csv')
#%%
    
DEGREE = [3]; K_FOLD = 5;
data = imp.standardize(imp.remove_outliers(
                            imp.outliers_to_mean(INPUT_DATA.copy())))
data_pr = imp.standardize(INPUT_DATA_PR.copy())
#Logistic regression
if log_regress:
    

    gammas = np.logspace(-11, -3.5, 20)
    lambda_ = 10
    max_iter = 100
     
    accuracy_poly_log_reg = np.zeros((len(DEGREE), gammas.shape[0]))
    for d in range(len(DEGREE)): 
        data_poly = imp.build_poly(data, DEGREE[d])
        y_tr, y_te, data_poly_tr, data_poly_te = help1.split_data(Y_IN, data_poly, K_FOLD)
        initial_w = np.ones(data_poly_tr.shape[2])
        ws_poly_log_reg = np.zeros((K_FOLD, data_poly_tr.shape[2])); losses_poly_log_reg = [];
        for g in range(gammas.shape[0]):
            
            for k in range(K_FOLD):
                w_temp, loss_temp = imp.logistic_regression(y_tr[k], data_poly_tr[k], initial_w, max_iter, gammas[g])
                w_temp, loss_temp = imp.reg_logistic_regression(y_tr[k], data_poly_tr[k], lambda_, initial_w, max_iter, gammas[g])
                ws_poly_log_reg[k] = np.array(w_temp)
                losses_poly_log_reg.append(loss_temp)
            w_poly_log_reg_mean = np.array(ws_poly_log_reg.mean(axis=0))
            accuracy = []
            for k in range(K_FOLD):
                accuracy.append(imp.calculate_accuracy(data_poly_te[k], y_te[k], w_poly_log_reg_mean))
            accuracy_poly_log_reg[d, g] = np.array(accuracy).mean()
            print('Accuracy for {} degree poly and gamma = {} is {}'.format(\
                  DEGREE[d], gammas[g], np.array(accuracy).mean()))
            
    temp_ind = np.where(accuracy_poly_log_reg == accuracy_poly_log_reg.max())
    best_acc_ind = (temp_ind[0][0],temp_ind[1][0])
    best_acc = accuracy_poly_log_reg[best_acc_ind]
    best_gamma = gammas[best_acc_ind[1]]
    best_deg = DEGREE[best_acc_ind[0]]
    print('the best accuracy is {} with degree {} and gamma {}'.format(\
            best_acc, best_deg, best_gamma))
    
    #generate figure of accuracy depending on gamma values for different polynomial degrees
    fig6 = plt.figure(6)
    ax6 = fig6.add_subplot(111)
    ax6.set_xscale('log')
    for d in range(len(DEGREE)):
        plt.plot(gammas, accuracy_poly_log_reg[d, :], label='Accuracy w/ poly deg: '+str(DEGREE[d])) 
    plt.title('Accuracy for logistic regression')
    plt.xlabel('gamma'); plt.ylabel('Accuracy');
    plt.legend(loc=0)
    plt.show()
        
    
#%%
    
#Ridge regression with poly for different lambdas
if POLY_RIDGE:
    
    data = imp.standardize(imp.remove_outliers(imp.outliers_to_mean(INPUT_DATA.copy())))
    data_pr = imp.standardize(INPUT_DATA_PR.copy())
    
    DEGREE = [6]; K_FOLD = 5; BEST = True
    if BEST:
        lambdas = np.array([2.6320240607048775e-06])
    else:
        lambdas = np.logspace(np.log10(1e-8), np.log10(1e-4), 70)
    
    accuracy_poly_ridge = np.zeros((len(DEGREE), lambdas.shape[0]))
    for d in range(len(DEGREE)):
        data_poly = imp.build_poly(data, DEGREE[d])
        y_tr, y_te, data_poly_tr, data_poly_te = help1.split_data(Y_IN, data_poly, K_FOLD)
        
        ws_poly_ridge = np.zeros((K_FOLD, data_poly_tr.shape[2])); losses_poly_ridge = []; 
        for l in range(lambdas.shape[0]):
            for k in range(K_FOLD):
                w_temp, loss_temp = imp.ridge_regression(y_tr[k], data_poly_tr[k], lambdas[l])
                ws_poly_ridge[k] = np.array(w_temp)
                losses_poly_ridge.append(loss_temp)
            w_poly_ridge_mean = np.array(ws_poly_ridge.mean(axis=0))
            accuracy = []
            for k in range(K_FOLD):
                accuracy.append(imp.calculate_accuracy(data_poly_te[k], y_te[k], w_poly_ridge_mean))
                
            accuracy_poly_ridge[d ,l] = np.array(accuracy).mean()
            print('Accuracy for {} degree poly and lambda = {} is {}'.format(\
                  DEGREE[d], lambdas[l], np.array(accuracy).mean()))
    temp_ind = np.where(accuracy_poly_ridge == accuracy_poly_ridge.max())
    best_acc_ind = (temp_ind[0][0],temp_ind[1][0])
    best_acc = accuracy_poly_ridge[best_acc_ind]
    best_lambda = lambdas[best_acc_ind[1]]
    best_deg = DEGREE[best_acc_ind[0]]
    print('the best accuracy is {} with degree {} and lambda {}'.format(\
            best_acc, best_deg, best_lambda))
    
    # Figure generation of accuracy depending on lambda values with different polynomial degrees    
    fig5 = plt.figure(3)
    ax5 = fig5.add_subplot(111)
    ax5.set_xscale('log')
    for d in range(len(DEGREE)):
        plt.plot(lambdas, accuracy_poly_ridge[d, :], label='Accuracy w/ poly deg: '+str(DEGREE[d])) 
    plt.title('Accuracy for polynomial reg')
    plt.xlabel('lambda'); plt.ylabel('Accuracy');
    plt.legend(loc=0)
    plt.show()


#%%
    
accuracy_reg = []; accuracy_off = [];  

# copies and cleans the data so we don't have to import each time
data_pr = imp.standardize(INPUT_DATA_PR.copy())
data = imp.remove_outliers(
        imp.standardize(imp.outliers_to_mean(INPUT_DATA.copy())));

    #Splits the data into k_fold 
K_FOLD = 5
y_tr, y_te, data_tr, data_te = help1.split_data(Y_IN, data, K_FOLD)

ws = []; losses = [];
for i in range(k_fold):
        # Loss values for all the data and cleaned data
        w, loss = imp.least_squares(y_tr[i], data_tr[i])
        ws.append(w); losses.append(loss) 
        
# Takes the mean w as the best approximation (test w/ median)
w_ls = np.array(ws).mean(axis=0)
accuracy_reg.append(imp.calculate_accuracy(data_te[0], y_te[0], w_ls))

# Same as before but with off set in the data
if OFFSET:
    data_off = np.c_[np.ones(data.shape[0]), data] 
    y_tr, y_te, data_tr_off, data_te_off = help1.split_data(Y_IN, data_off, k_fold)

    ws_off = []; losses_off = [];
    for i in range(k_fold):
            # Loss values for all the data and cleaned data
            w, loss = imp.least_squares(y_tr[i], data_tr_off[i])
            ws_off.append(w); losses_off.append(loss)

#%%

# Finds the accuracy        
w_ls_off = np.array(ws_off).mean(axis=0)
accuracy_off.append(imp.calculate_accuracy(data_te_off[0], y_te[0], w_ls_off))
        
if POLY:

    for d in range(len(DEGREE)):
        data_poly = imp.build_poly(data, DEGREE[d])
        y_tr, y_te, data_poly_tr, data_poly_te = help1.split_data(yb, data_poly, k_fold)
        
        ws_poly = []; losses_poly = [];
        for i in range(k_fold):
            w, loss = imp.least_squares(y_tr[i], data_poly_tr[i])
            ws_poly.append(w); losses_poly.append(loss)                    
        
        w_poly_mean = np.array(ws_poly).mean(axis=0)   
        accuracy_poly[e, d] = imp.calculate_accuracy(data_poly_te[i], y_te[i], w_poly_mean)
       
         

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

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
