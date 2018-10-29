# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from src import implementations as imp
from src import proj1_helpers as help1

DATA_PATH = './all/'
# Set to false after first run !
RELOAD_DATA = False
# Set to true to run specific method demo
GRADIENT_DESCENT = False
STOCHASTIC_GRADIENT_DESCENT = False
REG_LIN = False
OFFSET = False 
POLY = False
POLY_RIDGE = False
RIDGE_CAT = False
LOG_REGRESS = True
REG_LOG_REGRESS = False

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
            accuracy_grid_GD[d, g], _ = imp.cross_validation(imp.least_square_GD, y_tr, 
                                data_poly_tr, y_te, data_poly_te, k_fold=K_FOLD, 
                                initial_w=initial_w, max_iters=max_iters, gamma=gammas[g])
            
    temp_ind = np.where(accuracy_grid_GD == accuracy_grid_GD.max())
    best_acc_ind = (temp_ind[0][0],temp_ind[1][0])
    best_acc = accuracy_grid_GD[best_acc_ind]
    best_gamma = gammas[best_acc_ind[1]]
    best_deg = DEGREE[best_acc_ind[0]]
    print('the best accuracy is {} with degree {} and gamma {}'.format(
                                                best_acc, best_deg, best_gamma))

#%%  
if STOCHASTIC_GRADIENT_DESCENT:
    print('Stochastic Gradient descent:')
    #defining parameters. 
    DEGREE = [1, 2, 3]; K_FOLD = 5; gammas = np.logspace(-5, 0, 10); 
    max_iters = 150; batch_size = 10
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
            accuracy_grid_GD[d, g], _ = imp.cross_validation(imp.least_squares_SGD, y_tr, data_poly_tr, y_te, data_poly_te, k_fold=K_FOLD, initial_w=initial_w, max_iters=max_iters, gamma=gammas[g], batch_size=batch_size)
            
    temp_ind = np.where(accuracy_grid_GD == accuracy_grid_GD.max())
    best_acc_ind = (temp_ind[0][0],temp_ind[1][0])
    best_acc = accuracy_grid_GD[best_acc_ind]
    best_gamma = gammas[best_acc_ind[1]]
    best_deg = DEGREE[best_acc_ind[0]]
    print('the best accuracy is {} with degree {} and gamma {}'.format(
                                                best_acc, best_deg, best_gamma))
#%%
    

# Computes the weights for the linear regression with least squares method
if REG_LIN:  
    print('Linear regression (least squares) w/o offset:')
    accuracy_reg = []; K_FOLD = 5;
    # copies and cleans the data so we don't have to import each time
    data = imp.standardize(imp.remove_outliers(imp.outliers_to_mean(INPUT_DATA.copy())))
    data_pr = imp.standardize(imp.remove_outliers(imp.outliers_to_mean(INPUT_DATA_PR.copy())))    
    #Splits the data into k_fold 
    y_tr, y_te, data_tr, data_te = imp.split_data(Y_IN, data, K_FOLD)
    # Does the cross validation
    ws = []; losses = [];
    accuracy_reg, w_ls = imp.cross_validation(imp.least_squares, y_tr, data_tr, y_te, data_te)
    print("{:.4}% accurate for lin_reg w/out offset".format(np.array(accuracy_reg) * 100))
    
    y_pred_ls = help1.predict_labels(w_ls, data)
    help1.create_csv_submission(IDS_PR, y_pred_ls, "reg_lin.csv")    
    
# Same as before but with offset in the data
if OFFSET:
    print('Linear regression (least_squares) with offset:')
    accuracy_off = []; K_FOLD = 5;  
    # copies and cleans the data so we don't have to import each time
    data = imp.standardize(imp.remove_outliers(imp.outliers_to_mean(INPUT_DATA.copy())))
    data_pr = imp.standardize(imp.remove_outliers(imp.outliers_to_mean(INPUT_DATA_PR.copy())))   
    # Creates the data with the offset and splits it
    data_off = imp.build_poly(data, 1) 
    data_off_pr = imp.build_poly(data_pr, 1)
    y_tr, y_te, data_tr_off, data_te_off = imp.split_data(Y_IN, data_off, K_FOLD)
    # Does the cross validation
    accuracy_reg, w_ls_off = imp.cross_validation(imp.least_squares, y_tr, data_tr_off, y_te, data_te_off)

    print("{:.4}% accurate for lin_reg w/ offset".format(accuracy_reg * 100))
    y_pred_ls_off = help1.predict_labels(w_ls_off, data_off_pr)
    help1.create_csv_submission(IDS_PR, y_pred_ls_off, "reg_lin_off.csv")    
#%%

# Finds accuracy for different degree polynomial expension for least squares method         
if POLY:
    # Initialising parameters
    DEGREE = [1, 2, 3, 5, 6, 7, 9]; K_FOLD = 5;
    print("Polynomial regression:")
    data = imp.standardize(imp.remove_outliers(imp.outliers_to_mean(INPUT_DATA.copy())))
    data_pr = imp.standardize(imp.remove_outliers(imp.outliers_to_mean(INPUT_DATA_PR.copy())))  
    accuracy_poly = []; w_poly_mean = [];
    for d in range(len(DEGREE)):
        data_poly = imp.build_poly(data, DEGREE[d])
        y_tr, y_te, data_poly_tr, data_poly_te = imp.split_data(Y_IN, data_poly, K_FOLD)
        
        acc_poly, w_poly = imp.cross_validation(imp.least_squares, y_tr, data_poly_tr, y_te, data_poly_te)
        accuracy_poly.append(acc_poly); w_poly_mean.append(w_poly);
        print("{:.6} % accurate for polynomial_reg w/ deg: ".format(accuracy_poly[d] * 100) +str(DEGREE[d]))
        
    idmax = np.array(accuracy_poly).argmax()
    print("\n\nBest accuracy is achieved with deg: {}, {:.6} %".format(DEGREE[idmax], accuracy_poly[idmax] * 100))
    
    y_pred_poly = help1.predict_labels(w_poly_mean[idmax], imp.build_poly(data_pr, DEGREE[idmax]))
    help1.create_csv_submission(Y_IN_PR, y_pred_poly, "poly_reg-deg"+str(DEGREE[idmax])+".csv")
    
    
    # PLots the accuracy for both models so that we can find the best error threshold    
    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(111)
    plt.plot(DEGREE, accuracy_poly) 
    plt.title('Accuracy for polynomial reg')
    plt.xlabel('Degree of polynomial'); plt.ylabel('Accuracy');
    plt.show()   
    
#%%
    
#Ridge regression with poly for different lambdas
if POLY_RIDGE:
    
    # Treatment applied to the data
    print("Polynomial ridge_regression:", end='\n\n');
    data = imp.standardize(imp.remove_outliers(imp.outliers_to_mean(INPUT_DATA.copy())))
    data_pr = imp.standardize(imp.remove_outliers(imp.outliers_to_mean(INPUT_DATA_PR.copy())))
    # Initialisaes the parameters for the regression
    DEGREE = [9]; K_FOLD = 5;
    lambdas = np.logspace(np.log10(1e-9), np.log10(1e-1), 10)
    
    accuracy_poly_ridge = np.zeros((len(DEGREE), lambdas.shape[0]))
    for d in range(len(DEGREE)):
        data_poly = imp.build_poly(data, DEGREE[d])
        y_tr, y_te, data_poly_tr, data_poly_te = imp.split_data(Y_IN, data_poly, K_FOLD)
        for l in range(lambdas.shape[0]):
            
            accuracy_poly_ridge[d ,l], _ = imp.cross_validation(
                    imp.ridge_regression, y_tr, data_poly_tr,y_te, data_poly_te, lambda_=lambdas[l])
            
            print(str((d)*lambdas.shape[0] + (l+1))+'/'+str(lambdas.shape[0] * len(DEGREE)), \
                  'Accuracy for {} degree poly and lambda = {:.5E} is {:.6}'.format(
                  DEGREE[d], lambdas[l], accuracy_poly_ridge[d ,l]*100))
         
    temp_ind = np.where(accuracy_poly_ridge == accuracy_poly_ridge.max())
    best_acc_ind = (temp_ind[0][0],temp_ind[1][0])
    best_lambda = lambdas[best_acc_ind[1]]

    ws_best = []
    data_poly = imp.build_poly(data, DEGREE[best_acc_ind[0]])
    y_tr, y_te, data_poly_tr, data_poly_te = imp.split_data(Y_IN, data_poly, K_FOLD)
    _, w_poly_ridge_best = imp.cross_validation(imp.ridge_regression, 
                                                y_tr, data_poly_tr, y_te, data_poly_te, 
                                                lambda_= best_lambda)
    
    y_pred_ridge = help1.predict_labels(w_poly_ridge_best, imp.build_poly(data_pr, DEGREE[best_acc_ind[0]]))
    help1.create_csv_submission(IDS_PR, y_pred_ridge, "ridge_deg"+str(DEGREE[best_acc_ind[0]])+".csv")
        
    print('The best accuracy is {:.5} with degree {} and lambda {:.6}'.format(
            accuracy_poly_ridge[best_acc_ind], DEGREE[best_acc_ind[0]], best_lambda))
    
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
    data_cat, y_cat, data_cat_pr, y_cat_pr, ids_cat_pr = imp.split_by_category(
            INPUT_DATA.copy(), Y_IN.copy(), INPUT_DATA_PR.copy(), Y_IN_PR.copy(), IDS_PR.copy())
    # Parameters for ridge_cat
    DEGREE = [9]; K_FOLD = 5;
    lambdas = np.logspace(np.log10(1e-6), np.log10(1e-1), 10)
    
    ws_best_cat = []; accuracy_cat = np.zeros((len(y_cat), len(DEGREE), lambdas.shape[0]))
    w_cat_ridge_best = []; ind = [];
    y_pred_cat = [];
    for c in range(len(y_cat)):
        print("Computing w for case: "+str(c))
        data_cat[c] = imp.standardize(imp.remove_outliers(imp.outliers_to_mean(data_cat[c])))       
        data_cat_pr[c] = imp.standardize(imp.remove_outliers(imp.outliers_to_mean(data_cat_pr[c])))
        for d in range(len(DEGREE)):
            data_poly = imp.build_poly(data_cat[c], DEGREE[d])
            y_tr, y_te, data_poly_tr, data_poly_te = imp.split_data(y_cat[c], data_poly, K_FOLD)
            
            for l in range(lambdas.shape[0]):
                # Cross validation and determination of the accuracy
                accuracy_cat[c, d, l], _ = imp.cross_validation(imp.ridge_regression, 
                            y_tr, data_poly_tr, y_te, data_poly_te, lambda_=lambdas[l])
         
        idmax = np.matrix(accuracy_cat[c]).argmax()
        ind.append((int(idmax / accuracy_cat[0].shape[1]), int(idmax / accuracy_cat[0].shape[0])))
        w_temp, _ = imp.ridge_regression(y_cat[c], imp.build_poly(data_cat[c], DEGREE[ind[c][0]]), lambdas[ind[c][1]])
        w_cat_ridge_best.append(w_temp)
    
        # Does the prediction    
        if c == 0:
            y_pred = np.array(help1.predict_labels(w_cat_ridge_best[c], imp.build_poly(data_cat_pr[c], DEGREE[-1])))
            ids_pr = np.array(ids_cat_pr[c])
        else:            
            y_pred = np.r_[y_pred, help1.predict_labels(w_cat_ridge_best[c], imp.build_poly(data_cat_pr[c], DEGREE[-1]))]
            ids_pr = np.r_[ids_pr, ids_cat_pr[c]]
    
    help1.create_csv_submission(ids_pr, y_pred, "ridge_deg9_catsep.csv")        
    
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
   
#%%
#Logistic regression
if LOG_REGRESS:
    
    DEGREE = [1]; K_FOLD = 5;
    data = imp.normalize(imp.remove_outliers(
                            imp.outliers_to_mean(INPUT_DATA.copy())))

    gammas = np.logspace(-5, 0, 2)
    max_iter = 10
     
    accuracy_grid_logreg = np.zeros((len(DEGREE), gammas.shape[0]))
    for d in range(len(DEGREE)): 
        data_poly = imp.build_poly(data, DEGREE[d])
        y_tr, y_te, data_poly_tr, data_poly_te = imp.split_data(Y_IN, data_poly, K_FOLD)
        initial_w = np.ones(data_poly_tr.shape[2])
        ws_poly_log_reg = np.zeros((K_FOLD, data_poly_tr.shape[2])); losses_poly_log_reg = [];
        for g in range(gammas.shape[0]):    
            accuracy_grid_logreg[d,g], _ = imp.cross_validation(
                        imp.logistic_regression, y_tr, data_poly_tr, y_te, 
                        data_poly_te, gamma=gammas[g], 
                        initial_w = np.zeros(data_poly_tr.shape[2]))
    
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
        plt.plot(gammas, accuracy_grid_logreg[d, :], label='Accuracy w/ poly deg: '+str(DEGREE[d])) 
    plt.title('Accuracy for logistic regression')
    plt.xlabel('gamma'); plt.ylabel('Accuracy');
    plt.legend(loc=0)
    plt.show()
        
#%%
    
if REG_LOG_REGRESS:    

    gammas = np.logspace(-12, 0, 3)
    lambdas = np.logspace(-10, 0, 3)
    max_iter = 25
    DEGREE = [1]; K_FOLD = 5;
    data = imp.standardize(imp.remove_outliers(
                            imp.outliers_to_mean(INPUT_DATA.copy())))
    #creates a 3d array that stores the accuracy for each degree, gamma and lambda
    accuracy_grid_reglogreg = np.zeros((len(DEGREE), gammas.shape[0], lambdas.shape[0]))
    
    for d in range(len(DEGREE)): 
        data_poly = imp.build_poly(data, DEGREE[d])
        y_tr, y_te, data_poly_tr, data_poly_te = imp.split_data(Y_IN, data_poly, K_FOLD)
        ws_poly_log_reg = np.zeros((K_FOLD, data_poly_tr.shape[2])); losses_poly_log_reg = [];
        for g in range(gammas.shape[0]):
            for l in range(lambdas.shape[0]):
                accuracy_grid_reglogreg[d, g, l], _ = imp.cross_validation(imp.reg_logistic_regression, 
                    y_tr, data_poly_tr, y_te, data_poly_te, gamma=gammas[g], 
                    lambda_=lambdas[l], initial_w = np.random.rand(data_poly_tr.shape[2]))
                
    temp_ind = np.where(accuracy_grid_reglogreg == accuracy_grid_reglogreg.max())
    best_acc_ind = (temp_ind[0][0],temp_ind[1][0], temp_ind[2][0])
    best_acc = accuracy_grid_reglogreg[best_acc_ind]
    best_gamma = gammas[best_acc_ind[1]]
    best_lambda = lambdas[best_acc_ind[2]]
    best_deg = DEGREE[best_acc_ind[0]]
    
   #best_w, _ = imp.reg_logistic_regression(y_tr, data_poly_tr[k], lambdas[l], initial_w, max_iter, gammas[g])
    print('Best accuracy = {}; Degree = {}; Gamma = {}; Lambda {}'.format(\
            best_acc, best_deg, best_gamma, best_lambda))
    
    #generate figure of accuracy depending on gamma values for different polynomial degrees
    fig7 = plt.figure(7)
    ax7 = fig7.add_subplot(111)
    ax7.set_xscale('log')
    for g in range(len(gammas)):
        plt.plot(lambdas, accuracy_grid_reglogreg[best_acc_ind[0],g,:], label='Accuracy w/ gamma: '+str(gammas[g])) 
    plt.title('Accuracy for logistic regression')
    plt.xlabel('lambda'); plt.ylabel('Accuracy');
    plt.legend(loc=0)
    plt.show()
