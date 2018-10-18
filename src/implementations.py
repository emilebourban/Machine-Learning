import numpy as np

def calculate_mse(e):
    '''Calculates the Mean Square Error'''
    return 1/2 * np.mean(e**2)

def calculate_mae(e):
    ''' Calculates the Mean Absolute Error '''
    return np.mean(np.abs(e))
    

def compute_loss(y, tx, w, error='square', rmse=False):
    '''  Computes loss function, type=square/absolute'''
    e = y - (tx @ w)
    if type == 'absolute':
        return calculate_mae(e)
    elif rmse:
        return np.sqrt(2 * calculate_mse(e))
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
    
def ridge_regression(y, tx, lambda_):
	'''Ridge regression using normal equations'''
    A = tx.T.dot(tx) - lambda_/(2*len(y))*np.identity(len(y))
    B = tx.T.dot(y) 
    w = np.linalg.solve(A, B)   
    loss = compute_loss(y, tx, w, rmse=True)
    
    return (w, loss)

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

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






def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)




def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    copy_k_indices = list(k_indices.copy())
    test_indices = copy_k_indices[k]
    test_y = [y[v] for v in test_indices]
    test_x = [x[v] for v in test_indices]
    del copy_k_indices[k]
    train_indices = [v for t in copy_k_indices for v in t] 
    train_y = [y[v] for v in train_indices]
    train_x = [x[v] for v in train_indices]
    train_poly = build_poly(train_x,degree)
    test_poly = build_poly(test_x, degree)
    weights = ridge_regression(train_y,train_poly,lambda_)
    loss_tr = compute_mse(train_y, train_poly, weights)
    loss_te = compute_mse(test_y, test_poly, weights)
    return loss_tr, loss_te



from plots import cross_validation_visualization# 

def cross_validation_demo(y,x):
    seed = 1
    degree = 7
    k_fold = 4
    lambdas = np.logspace(-4, 0, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # ***************************************************
    print(lambdas)
    for l in lambdas :
        rmse_tr_0 = 0
        rmse_te_0 = 0
        for k in range(k_fold):
            o,p = cross_validation(y, x, k_indices,k,l, degree )
            rmse_tr_0 += o
            rmse_te_0 += p 
        rmse_tr.append(rmse_tr_0/4)
        rmse_te.append(rmse_te_0/4)
    # ***************************************************  
    print(len(x))
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)



def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)

        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws


def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")



