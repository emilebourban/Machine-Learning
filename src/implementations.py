import numpy as np

def compute_mse(y, tx, w):
    '''Calculates the Mean Square Error'''
    return 1/2 * np.mean(e**2)


def calculate_mae(e):
    ''' Calculates the Mean Absolute Error '''
    return np.mean(np.abs(e))
    

def compute_loss(y, tx, w, error='square'):
    '''  Computes loss function, type=square/absolute'''
    e = y - (tx @ w)
    if type == 'absolute':
        return calculate_mae(e)
    else:
        return calculate_mse(e)
    

#def least_square_GD(y, tx, initial_w, max_iters, gamma):
#	'''Linear regression using gradient descent'''	
#    
#    return (w, loss)
#   
#
#
#
#def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
#	'''Linear regression using stochastic gradient descent'''
#	
#   return (w, loss) 
    
        
def least_squares(y, tx):
    """calculate the least squares solution."""
    
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return (w, loss)
    
#def ridge_regression(y, tx, lambda_):
#	'''Ridge regression using normal equations'''
#    return (w, loss)
#    
#def logistic_regression(y, ty, initial_w, max_iters, gamma):
#	'''Logistic regression using gradient descent or SGD'''
#    return (w, loss)
    
#def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
#	'''Regularized logistic regression using gradient descent
#	or SGD'''
#	
#	
#    return (w, loss)
    
