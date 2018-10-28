# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from src import implementations as imp
from src import proj1_helpers as help1

DATA_PATH = './all/'
SUB_NAME = 'sample_submission.csv'

# Imports the data
Y_IN, INPUT_DATA, IDS = help1.load_csv_data(DATA_PATH+r'train.csv')
Y_IN_PR, INPUT_DATA_PR, IDS_PR = help1.load_csv_data(DATA_PATH+r'test.csv')

# Seporates the data into 4 matrix according to the category given by col 22
data_cat, y_cat, data_cat_pr, y_cat_pr, ids_cat_pr = imp.split_by_category(INPUT_DATA.copy(), 
                                                                           Y_IN.copy(), 
                                                                           INPUT_DATA_PR.copy(), 
                                                                           Y_IN_PR.copy(), 
                                                                           IDS_PR.copy())
# Sets the parameters for the regression
DEGREE = [9]; K_FOLD = 5;
lambdas = np.logspace(np.log10(1e-6), np.log10(1e-1), 100)

# Initiation of all the empty variables that we will need later
ws_best_cat = []; accuracy_cat = np.zeros((len(y_cat), len(DEGREE), lambdas.shape[0]))
w_cat_ridge_best = []; ind = []; y_pred_cat = []
for c in range(len(y_cat)):
    print("Computing w for case: "+str(c))
    # Standardisation and removal of the ouliers in the data and prediction data
    data_cat[c] = imp.standardize(imp.remove_outliers(imp.outliers_to_mean(data_cat[c])))       
    data_cat_pr[c] = imp.standardize(imp.remove_outliers(imp.outliers_to_mean(data_cat_pr[c])))
    for d in range(len(DEGREE)):
        # Creation of polynomial matrix of factors for regression and spliting of the data
        data_poly = imp.build_poly(data_cat[c], DEGREE[d])
        y_tr, y_te, data_poly_tr, data_poly_te = imp.split_data(y_cat[c], data_poly, K_FOLD)
        # Iteration on the lambda logspace
        for l in range(lambdas.shape[0]):
            ws_poly_ridge = [];
            # Cross validation
            for k in range(K_FOLD):
                w_temp, _ = imp.ridge_regression(y_tr[k], data_poly_tr[k], lambdas[l])
                ws_poly_ridge.append(w_temp)
            # Mean w as the best approximation
            w_poly_ridge_mean = np.array(ws_poly_ridge).mean(axis=0)
            
            # Computes the accuracy to find the best parameters later
            accuracy = [];
            for k in range(K_FOLD):
                accuracy.append(imp.calculate_accuracy(data_poly_te[k], y_te[k], w_poly_ridge_mean))                    
            accuracy_cat[c, d , l] = np.array(accuracy).mean()
#TODO Fix step display       
        print('Step: '+str(c*(len(y_cat)+len(DEGREE)+lambdas.shape[0]))+'/'
                           +str(lambdas.shape[0] * len(DEGREE) * len(y_cat)) )
        
    # Captures the index of the parameters for the best accuracy
    idmax = np.matrix(accuracy_cat[c]).argmax()
    ind.append((int(idmax / accuracy_cat[0].shape[1]), int(idmax / accuracy_cat[0].shape[0])))
    # Computes the w for each category with the best parameters and stores it
    w_temp, _ = imp.ridge_regression(y_cat[c], imp.build_poly(data_cat[c], DEGREE[-1]), lambdas[ind[c][1]])
    w_cat_ridge_best.append(w_temp)
    
    # Computation of the prediction keeping the y_pred and the ids in the same order
    if c == 0:
        y_pred = np.array(help1.predict_labels(w_cat_ridge_best[c], imp.build_poly(data_cat_pr[c], DEGREE[-1])))
        ids_pr = np.array(ids_cat_pr[c])
    else:            
        y_pred = np.r_[y_pred, help1.predict_labels(w_cat_ridge_best[c], imp.build_poly(data_cat_pr[c], DEGREE[-1]))]
        ids_pr = np.r_[ids_pr, ids_cat_pr[c]]
        
# Plots the accuracy for each category of data in function of lambda
fig5 = plt.figure(3)
ax5 = fig5.add_subplot(111)
ax5.set_xscale('log')
for c in range(len(y_cat)):
    for d in range(len(DEGREE)):
        plt.plot(lambdas, accuracy_cat[c, d, :], label='Accuracy w/ poly deg: '+str(DEGREE[d])+', cat: '+str(c)) 
plt.title('Accuracy for ridge reg with var separation')
plt.xlabel('lambda'); plt.ylabel('Accuracy');
plt.legend(loc=0)
plt.show() 

# Creates the submission file
help1.create_csv_submission(ids_pr, y_pred, SUB_NAME)












