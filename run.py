# -*- coding: utf-8 -*-

import numpy as np
from src import implementations as imp
from src import proj1_helpers as help1

DATA_PATH = './all/'
SUB_NAME = 'sample_submission.csv'

# Imports the data
Y_IN, INPUT_DATA, IDS = help1.load_csv_data(DATA_PATH+r'train.csv')
Y_IN_PR, INPUT_DATA_PR, IDS_PR = help1.load_csv_data(DATA_PATH+r'test.csv')

data = imp.standardize(imp.outliers_to_zero(INPUT_DATA.copy()))
data_pr = imp.standardize(INPUT_DATA_PR.copy())

# Used the berst method we currently have to predict the weights
DEGREE = 6; K_FOLD = 5;

data_poly = imp.build_poly(data, DEGREE)
y_tr, y_te, data_poly_tr, data_poly_te = help1.split_data(Y_IN, data_poly, K_FOLD)

ws_poly = []; losses_poly = [];
for i in range(K_FOLD):
    w, loss = imp.least_squares(y_tr[i], data_poly_tr[i])
    ws_poly.append(w); losses_poly.append(loss)                    

w_poly_mean = np.array(ws_poly).mean(axis=0)   
accuracy_poly = [];
for i in range(K_FOLD):
    accuracy_poly.append(imp.calculate_accuracy(data_poly_te[i], y_te[i], w_poly_mean))

print("{} % accurate".format(np.array(accuracy_poly).mean()))

data_poly_pr = imp.build_poly(data_pr, DEGREE)
y_pred = help1.predict_labels(w_poly_mean, data_poly_pr)
help1.create_csv_submission(IDS_PR, y_pred, SUB_NAME)













