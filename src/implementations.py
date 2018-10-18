import numpy as np

def calculate_mse(e):
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
    

def least_square_GD(y, tx, initial_w, max_iters, gamma):
    '''Linear regression using gradient descent'''	
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return ws[-1], loss[-1]

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    print(y.shape, tx.shape, w.shape)
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    '''Linear regression using stochastic gradient descent'''
    return (w, loss) 
    
        
def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss
    
#def ridge_regression(y, tx, lambda_):
#	'''Ridge regression using normal equations'''
#    return (w, loss)
#    
#def logistic_regression(y, ty, initial_w, max_iters, gamma):
#	'''Logistic regression using gradient descent or SGD'''
#    return (w, loss)
#    
#def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
#	'''Regularized logistic regression using gradient descent
#	or SGD'''
#	
#	
#    return (w, loss)
