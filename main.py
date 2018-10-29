# -*- coding: utf-8 -*-
import datetime
import numpy as np
import matplotlib.pyplot as plt
from src import implementations as imp
from src import proj1_helpers as help1

DATA_PATH = './all/'
# Set to false after first run !
RELOAD_DATA = False
# Set to true to run specific method demo
GRADIENT_DESCENT = False
LOG_REGRESS = False
REG_LOG_REGRESS = False
POLY_RIDGE = True
RIDGE_CAT = True
REG_LIN = False
OFFSET = False 
POLY = False


#%%

# Only reload once, takes a lot of time
if RELOAD_DATA:
    print("Importing...")
    Y_IN, INPUT_DATA, IDS = help1.load_csv_data(DATA_PATH+r'train.csv')
    Y_IN_PR, INPUT_DATA_PR, IDS_PR = help1.load_csv_data(DATA_PATH+r'test.csv')
    
#%%  
if GRADIENT_DESCENT:
    print('Gradient descent:')
    #defining parameters. 
    DEGREE = [1, 2, 3]; K_FOLD = 5; gammas = np.logspace(-5, 0, 30); max_iters = 50
    #data standardization and removing of missing value / outliers
    data = imp.standardize(imp.remove_outliers(
                            imp.outliers_to_mean(INPUT_DATA.copy())))
    data_pr = imp.standardize(INPUT_DATA_PR.copy())
    
    accuracy_grid_GD = np.zeros((len(DEGREE), gammas.shape[0]))
    for d in range(len(DEGREE)):
        data_poly = imp.build_poly(data, DEGREE[d])
        y_tr, y_te, data_poly_tr, data_poly_te = imp.split_data(Y_IN, data_poly, K_FOLD)
        initial_w = np.ones(data_poly_tr.shape[2])
        for g in range(gammas.shape[0]):
            accuracy_grid_GD[d, g] = imp.cross_validation(imp.least_square_GD, y_tr, 
                                data_poly_tr, y_te, data_poly_te, k_fold=K_FOLD, 
                                initial_w=initial_w, max_iters=max_iters, gamma=gammas[g])
            
    temp_ind = np.where(accuracy_grid_GD == accuracy_grid_GD.max())
    best_acc_ind = (temp_ind[0][0],temp_ind[1][0])
    best_acc = accuracy_grid_GD[best_acc_ind]
    best_gamma = gammas[best_acc_ind[1]]
    best_deg = DEGREE[best_acc_ind[0]]
    print('the best accuracy is {} with degree {} and gamma {}'.format(
                                                best_acc, best_deg, best_gamma))
    
#TODO: calculate time for grad descent and compare with least squares
 
#%%
    
K_FOLD = 5
# Computes the weights for the linear regression with least squares method
if REG_LIN:  
    print('Linear regression (least squares) w/o offset:')
    accuracy_reg = []    
    # copies and cleans the data so we don't have to import each time
    data = imp.standardize(imp.remove_outliers(imp.outliers_to_mean(INPUT_DATA.copy())))
    data_pr = imp.standardize(imp.remove_outliers(imp.outliers_to_mean(INPUT_DATA_PR.copy())))    
    #Splits the data into k_fold 
    y_tr, y_te, data_tr, data_te = imp.split_data(Y_IN, data, K_FOLD)
    # Does the cross validation
    ws = []; losses = [];
#    for i in range(K_FOLD):
#            w, _ = imp.least_squares(y_tr[i], data_tr[i])
#            ws.append(w);    
       
    # Takes the mean w as the best approximation (test w/ median)
#    w_ls = np.array(ws).mean(axis=0)
#    accuracy_reg.append(imp.calculate_accuracy(data_te[0], y_te[0], w_ls))
    accuracy_reg = imp.cross_validation(imp.least_squares, y_tr, data_tr, y_te, data_te)
    print("{:.4}% accurate for lin_reg w/out offset".format(np.array(accuracy_reg) * 100))
    
    y_pred_ls = help1.predict_labels(w_ls, data)
    help1.create_csv_submission(IDS_PR, y_pred_ls, "reg_lin.csv")    
    
# Same as before but with off set in the data
if OFFSET:
    print('Linear regression (least_squares) with offset:')
    accuracy_off = [];   
    # copies and cleans the data so we don't have to import each time
    data = imp.standardize(imp.remove_outliers(imp.outliers_to_mean(INPUT_DATA.copy())))
    data_pr = imp.standardize(imp.remove_outliers(imp.outliers_to_mean(INPUT_DATA_PR.copy())))   
    # Creates the data with the offset and splits it
    data_off = np.c_[np.ones(data.shape[0]), data] 
    data_off_pr = np.c_[np.ones(data_pr.shape[0]), data_pr] 
    y_tr, y_te, data_tr_off, data_te_off = imp.split_data(Y_IN, data_off, K_FOLD)
    # Does the cross validation
#    ws_off = []; losses_off = [];
#    for i in range(K_FOLD):
#            w, _ = imp.least_squares(y_tr[i], data_tr_off[i])
#            ws_off.append(w);
#    w_ls_off = np.array(ws_off).mean(axis=0)
#    accuracy_off.append(imp.calculate_accuracy(data_te_off[0], y_te[0], w_ls_off))
    accuracy_reg = imp.cross_validation(imp.least_squares, y_tr, data_tr, y_te, data_te)

    print("{:.4}% accurate for lin_reg w/ offset".format(accuracy_reg * 100))
    y_pred_ls = help1.predict_labels(w_ls_off, data_off)
    help1.create_csv_submission(IDS_PR, y_pred_ls, "reg_lin_off.csv")    
#%%

# Finds accuracy for different degree polynomial expension for least squares method         
if POLY:
    # Initialising parameters
    DEGREE = range(22); K_FOLD = 5;
    print("Polynomial regression:"); print("Cleaning the data...")
    data = imp.standardize(imp.remove_outliers(imp.outliers_to_mean(INPUT_DATA.copy())))
    data_pr = imp.standardize(imp.remove_outliers(imp.outliers_to_mean(INPUT_DATA_PR.copy())))  
    accuracy_poly = [];
    for d in range(len(DEGREE)):
        data_poly = imp.build_poly(data, DEGREE[d])
        y_tr, y_te, data_poly_tr, data_poly_te = imp.split_data(Y_IN, data_poly, K_FOLD)
        
#        ws_poly = []; accuracy = [];
#        for i in range(K_FOLD):
#            w_poly, _ = imp.least_squares(y_tr[i], data_poly_tr[i])
#            ws_poly.append(w_poly)
#            accuracy.append(imp.calculate_accuracy(data_poly_te[i], y_te[i], w_poly))
#        w_poly_mean = np.array(ws_poly).mean(axis=0)   
#        accuracy_poly.append(np.array(accuracy).mean())
        
        accuracy_poly.append(imp.cross_validation(imp.least_squares, y_tr, data_poly_tr, y_te, data_poly_te))
        #print("{:.6} % accurate for polynomial_reg w/ deg: ".format(accuracy_poly * 100) +str(DEGREE[d]))
        
    #print("\n\nBest accuracy is achieved with deg: {}, {:.6} %".format(DEGREE[idmax], accuracy_poly * 100))
#TODO
#    y_pred_poly = help1.predict_labels(w_poly_mean[idmax], imp.build_poly(data_pr, DEGREE[idmax]))
#    help1.create_csv_submission(Y_IN, y_pred_poly, "poly_reg-deg"+str(DEGREE[idmax])+".csv")
    
    
    # PLots the accuracy for both models so that we can find the best error threshold    
    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(111)
    plt.plot(DEGREE, accuracy_poly) 
    plt.title('Accuracy for polynomial reg')
    plt.xlabel('Degree of polynomial'); plt.ylabel('Accuracy');
#    for i in range(len(DEGREE)):
#        ax3.annotate(str(accuracy_poly[i]), xy=(DEGREE[i], accuracy_poly[i]))
    plt.show()   
    
#%%
    
#Ridge regression with poly for different lambdas
if POLY_RIDGE:
    
    # Treatment applied to the data
    print("Polynomial ridge_regression:", end='\n\n'); print("Standardisation...");
    data = imp.standardize(imp.remove_outliers(imp.outliers_to_mean(INPUT_DATA.copy())))
    data_pr = imp.standardize(imp.remove_outliers(imp.outliers_to_mean(INPUT_DATA_PR.copy())))
    # Initialisaes the parameters for the regression
    DEGREE = [9]; K_FOLD = 5;
    lambdas = np.logspace(np.log10(1e-9), np.log10(1e-1), 10)
    
    accuracy_poly_ridge = np.zeros((len(DEGREE), lambdas.shape[0]))
    for d in range(len(DEGREE)):
        data_poly = imp.build_poly(data, DEGREE[d])
        y_tr, y_te, data_poly_tr, data_poly_te = imp.split_data(Y_IN, data_poly, K_FOLD)
        
        ws_poly_ridge = np.zeros((K_FOLD, data_poly_tr.shape[2])); losses_poly_ridge = []; 
        for l in range(lambdas.shape[0]):
#            for k in range(K_FOLD):
#                w_temp, loss_temp = imp.ridge_regression(y_tr[k], data_poly_tr[k], lambdas[l])
#                ws_poly_ridge[k] = np.array(w_temp)
#                losses_poly_ridge.append(loss_temp)
#            w_poly_ridge_mean = np.array(ws_poly_ridge.mean(axis=0))
#            accuracy = []
#            for k in range(K_FOLD):
#                accuracy.append(imp.calculate_accuracy(data_poly_te[k], y_te[k], w_poly_ridge_mean))
                
            accuracy_poly_ridge[d ,l] = imp.cross_validation(imp.ridge_regression, y_tr, data_poly_tr,y_te, data_poly_te, lambda_=lambdas[l])
            print(str((d)*lambdas.shape[0] + (l+1))+'/'+str(lambdas.shape[0] * len(DEGREE)), \
                  'Accuracy for {} degree poly and lambda = {:.5E} is {:.6}'.format(
                  DEGREE[d], lambdas[l], accuracy_poly_ridge[d ,l]*100))
#TODO same as category ridge           
    temp_ind = np.where(accuracy_poly_ridge == accuracy_poly_ridge.max())
    best_acc_ind = (temp_ind[0][0],temp_ind[1][0])
    best_acc = accuracy_poly_ridge[best_acc_ind]
    best_lambda = lambdas[best_acc_ind[1]]
    best_deg = DEGREE[best_acc_ind[0]]
    
    ws_best = []
    data_poly = imp.build_poly(data, best_deg)
    y_tr, y_te, data_poly_tr, data_poly_te = imp.split_data(Y_IN, data_poly, K_FOLD)
#    for k in range(K_FOLD):        
#        w_temp, _ = imp.ridge_regression(y_tr[k], data_poly_tr[k], best_lambda)
#        ws_best.append(w_temp)
#    w_poly_ridge_best = np.array(ws_best).mean(axis=0)
    w_poly_ridge_best = imp.cross_validation(imp.ridge_regression, y_tr, data_poly_tr, y_te, data_poly_te, lambda_= best_lambda)
            
    print('The best accuracy is {:.5} with degree {} and lambda {:.6}'.format(
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
    
if RIDGE_CAT:
    print("Polynomial regression with category splitting: ");
    data_cat, y_cat, data_cat_pr, y_cat_pr, ids_cat_pr = imp.split_by_category(INPUT_DATA.copy(), Y_IN.copy(), INPUT_DATA_PR.copy(), Y_IN_PR.copy(), IDS_PR.copy())
    # Parameters for ridge_cat
    DEGREE = [9]; K_FOLD = 5;
    lambdas = np.logspace(np.log10(1e-6), np.log10(1e-1), 100)
    
    ws_best_cat = []; accuracy_cat = np.zeros((len(y_cat), len(DEGREE), lambdas.shape[0]))
    for c in range(len(y_cat)):
        print("Computing w for case: "+str(c))
        data_cat[c] = imp.standardize(imp.remove_outliers(imp.outliers_to_mean(data_cat[c])))       
        data_cat_pr[c] = imp.standardize(imp.remove_outliers(imp.outliers_to_mean(data_cat_pr[c])))
        for d in range(len(DEGREE)):
            data_poly = imp.build_poly(data_cat[c], DEGREE[d])
            y_tr, y_te, data_poly_tr, data_poly_te = imp.split_data(y_cat[c], data_poly, K_FOLD)
            
            for l in range(lambdas.shape[0]):
#                ws_poly_ridge = np.zeros((K_FOLD, data_poly_tr.shape[2]));
#                for k in range(K_FOLD):
#                    w_temp, _ = imp.ridge_regression(y_tr[k], data_poly_tr[k], lambdas[l])
#                    ws_poly_ridge[k] = np.array(w_temp)
#                w_poly_ridge_mean = np.array(ws_poly_ridge).mean(axis=0)
#                
#                accuracy = []
#                for k in range(K_FOLD):
#                    accuracy.append(imp.calculate_accuracy(data_poly_te[k], y_te[k], w_poly_ridge_mean))                    
                accuracy_cat[c, d, l] = imp.cross_validation(imp.ridge_regression, y_tr, data_poly_tr, y_te, data_poly_te, lambda_=lambdas[l])
                
#                print(str((d)*lambdas.shape[0] + (l+1))+'/'+str(lambdas.shape[0] * len(DEGREE)), \
#                      'Accuracy for {} degree poly and lambda = {:.5E} is {:.6}'.format(
#                      DEGREE[d], lambdas[l], np.array(accuracy).mean()*100))
    w_cat_ridge_best = []; ind = []        
    for c in range(len(y_cat)):
        idmax = np.matrix(accuracy_cat[c]).argmax()
        ind.append((int(idmax / accuracy_cat[0].shape[1]), int(idmax / accuracy_cat[0].shape[0])))
        w_temp, _ = imp.ridge_regression(y_cat[c], imp.build_poly(data_cat[c], DEGREE[-1]), lambdas[ind[c][1]])
        w_cat_ridge_best.append(w_temp)
          
    fig5 = plt.figure(3)
    ax5 = fig5.add_subplot(111)
    ax5.set_xscale('log')
    for c in range(len(y_cat)):
        for d in range(len(DEGREE)):
            plt.plot(lambdas, accuracy_cat[c, d, :], label='Accuracy w/ poly deg: '+str(DEGREE[d])+', cat: '+str(c)) 
    plt.title('Accuracy for polynomial reg')
    plt.xlabel('lambda'); plt.ylabel('Accuracy');
    plt.legend(loc=0)
    plt.show()
    
    
    # Does the prediction
    y_pred_cat = []
    for c in range(4):
        if c == 0:
            y_pred = np.array(help1.predict_labels(w_cat_ridge_best[c], imp.build_poly(data_cat_pr[c], DEGREE[-1])))
            ids_pr = np.array(ids_cat_pr[c])
        else:            
            y_pred = np.r_[y_pred, help1.predict_labels(w_cat_ridge_best[c], imp.build_poly(data_cat_pr[c], DEGREE[-1]))]
            ids_pr = np.r_[ids_pr, ids_cat_pr[c]]
    
    help1.create_csv_submission(ids_pr, y_pred, "ridge_deg9_catsep.csv")        

   
#%%
#Logistic regression
if LOG_REGRESS:
    
    DEGREE = [1]; K_FOLD = 5;
    data = imp.normalize(imp.remove_outliers(
                            imp.outliers_to_mean(INPUT_DATA.copy())))
    data_pr = imp.standardize(INPUT_DATA_PR.copy())

    gammas = np.logspace(-11, 0, 10)
    max_iter = 100
     
    accuracy_grid_logreg = np.zeros((len(DEGREE), gammas.shape[0]))
    for d in range(len(DEGREE)): 
        data_poly = imp.build_poly(data, DEGREE[d])
        y_tr, y_te, data_poly_tr, data_poly_te = imp.split_data(Y_IN, data_poly, K_FOLD)
        initial_w = np.ones(data_poly_tr.shape[2])
        ws_poly_log_reg = np.zeros((K_FOLD, data_poly_tr.shape[2])); losses_poly_log_reg = [];
        for g in range(gammas.shape[0]):
            
#            for k in range(K_FOLD):
#                w_temp, loss_temp = imp.logistic_regression(y_tr[k], data_poly_tr[k], initial_w, max_iter, gammas[g])
#                w_temp, loss_temp = imp.logistic_regression(y_tr[k], data_poly_tr[k], initial_w, max_iter, gammas[g])
#                ws_poly_log_reg[k] = np.array(w_temp)
#                losses_poly_log_reg.append(loss_temp)
#            w_poly_log_reg_mean = np.array(ws_poly_log_reg.mean(axis=0))
#            accuracy = []
#            for k in range(K_FOLD):
#                accuracy.append(imp.calculate_accuracy(data_poly_te[k], y_te[k], w_poly_log_reg_mean))
#            accuracy_poly_log_reg[d, g] = np.array(accuracy).mean()
#            
            accuracy_grid_logreg[d,g] = imp.cross_validation(
                    imp.reg_logistic_regression, y_tr, data_poly_tr, y_te, 
                    data_poly_te, gamma=gammas[g], 
                    initial_w = np.zeros(data_poly_tr.shape[2])).mean()
    
    temp_ind = np.where(accuracy_grid_logreg == accuracy_grid_logreg.max())
    best_acc_ind = (temp_ind[0][0],temp_ind[1][0])
    best_acc = accuracy_grid_logreg[best_acc_ind]
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
    
if REG_LOG_REGRESS:
    

    gammas = np.logspace(-3, 2, 10)
    lambdas = np.logspace(-10, 1, 4)
    max_iter = 100
    
    #creates a 3d array that stores the accuracy for each degree, gamma and lambda
    accuracy_grid_reglogreg = np.zeros((len(DEGREE), gammas.shape[0], lambdas.shape[0]))
    
    for d in range(len(DEGREE)): 
        data_poly = imp.build_poly(data, DEGREE[d])
        y_tr, y_te, data_poly_tr, data_poly_te = imp.split_data(Y_IN, data_poly, K_FOLD)
        initial_w = np.ones(data_poly_tr.shape[2])
        ws_poly_log_reg = np.zeros((K_FOLD, data_poly_tr.shape[2])); losses_poly_log_reg = [];
        for g in range(gammas.shape[0]):
            for l in range(lambdas.shape[0]):
#                for k in range(K_FOLD):
#                    w_temp, loss_temp = imp.logistic_regression(y_tr[k], data_poly_tr[k], initial_w, max_iter, gammas[g])
#                    w_temp, loss_temp = imp.reg_logistic_regression(y_tr[k], data_poly_tr[k], lambdas[l], initial_w, max_iter, gammas[g])
#                    ws_poly_log_reg[k] = np.array(w_temp)
#                    losses_poly_log_reg.append(loss_temp)
#                w_poly_log_reg_mean = np.array(ws_poly_log_reg.mean(axis=0))
#                accuracy = []
#                for k in range(K_FOLD):
#                    accuracy.append(imp.calculate_accuracy(data_poly_te[k], y_te[k], w_poly_log_reg_mean))
#                accuracy_poly_log_reg[d, g, l] = np.array(accuracy).mean()
#                print('Accuracy = {:.4}; Degree = {}; Lambda = {:.3E}; Gamma = {:.3E}; Lambda*Gamma = {:.3} '.format(
#                      np.array(accuracy).mean(), DEGREE[d], lambdas[l], gammas[g], lambdas[l]*gammas[g]))
                accuracy_grid_reglogreg = imp.cross_validation(imp.reg_logistic_regression, y_tr, data_poly_tr, y_te, 
                    data_poly_te, gamma=gammas[g], lambda_=lambdas[l],
                    initial_w = np.zeros(data_poly_tr.shape[2])).mean()
    
    temp_ind = np.where(accuracy_grid_reglogreg == accuracy_grid_reglogreg.max())
    best_acc_ind = (temp_ind[0][0],temp_ind[1][0], temp_ind[2][0])
    best_acc = accuracy_grid_reglogreg[best_acc_ind]
    best_gamma = gammas[best_acc_ind[1]]
    best_lambda = lambdas[best_acc_ind[2]]
    best_deg = DEGREE[best_acc_ind[0]]
    
    best_w, _ = imp.reg_logistic_regression(y_tr[k], data_poly_tr[k], lambdas[l], initial_w, max_iter, gammas[g])
    print('Best accuracy = {}; Degree = {}; Gamma = {}; Lambda {}'.format(\
            best_acc, best_deg, best_gamma, best_lambda))
    
    #generate figure of accuracy depending on gamma values for different polynomial degrees
    fig7 = plt.figure(7)
    ax7 = fig7.add_subplot(111)
    ax7.set_xscale('log')
    for g in range(len(gammas)):
        plt.plot(lambdas, accuracy_grid_reglogreg[3,g,:], label='Accuracy w/ gamma: '+str(gammas[g])) 
    plt.title('Accuracy for logistic regression')
    plt.xlabel('gamma'); plt.ylabel('Accuracy');
    plt.legend(loc=0)
    plt.show()
    




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
