import numpy as np

def calculate_mse(e):
    '''Calculates the Mean Square Error'''
    return 1/2 * np.mean(e**2)

def calculate_mae(e):
    ''' Calculates the Mean Absolute Error '''
    return np.mean(np.abs(e))

def calculate_rmse(mse):
    '''Calculate root mean square error from mean square error''' 
    return np.sqrt(2 * mse)    

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
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
    return w, loss

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size):
    '''Linear regression using stochastic gradient descent'''
    w = initial_w 
    for i in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            grad = compute_gradient(minibatch_y, minibatch_tx, w)[0]
            w = w - gamma*grad
        loss = compute_loss(y, tx, w)
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
    A = tx.T.dot(tx) - (2*tx.shape[0]*lambda_)*np.identity(tx.shape[1])
    B = tx.T.dot(y) 
    w = np.linalg.solve(A, B)   
    loss = compute_loss(y, tx, w)
    
    return (w, loss)

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((x.shape[0], 1))
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


def calculate_accuracy(x, y, w):
    '''calculate the accuracy of a prediction w on a test set'''
    prediction = np.sign(x @ w) #gives the prediction of the model: negative values are taken as -1 and positive as +1
    comparison = (prediction == y)
    accuracy = sum(comparison) / y.shape[0]
    return accuracy


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


def remove_outliers(data):
    '''For each column, replaces the values that are further than 3 times the variance 
    from the mean by the mean.'''
    means = np.mean(data, axis=0)
    variances = np.std(data, axis=0)
    counts = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i,j] < (means[j] - 3 * variances[j]) or data[i,j] > (means[j] + 3 * variances[j]):
                data[i, j] = means[j]
                counts += 1
    print('{} outliers removed from data'.format(counts), end='\n\n') 
    return data
    
def outliers_to_zero(data):
    for i in range(data.shape[1]):
        col_mean = data[data[:, i] != -999, i].mean(axis=0)
        data[data[:, i] == -999, i] = np.ones(data[data[:, i] == -999, i].shape[0])*col_mean
    return data

def standardize(data):
    st_data = np.zeros((data.shape[0], data.shape[1]))
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    for i in range(data.shape[1]):
       st_data[:, i] = (data[:, i] - data_mean[i])/data_std[i]
        
    return st_data


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


def princomp(A):
 """ performs principal components analysis 
     (PCA) on the n-by-p data matrix A
     Rows of A correspond to observations, columns to variables. 

 Returns :  
  coeff :
    is a p-by-p matrix, each column containing coefficients 
    for one principal component.
  score : 
    the principal component scores; that is, the representation 
    of A in the principal component space. Rows of SCORE 
    correspond to observations, columns to components.
  latent : 
    a vector containing the eigenvalues 
    of the covariance matrix of A.
 """
 # computing eigenvalues and eigenvectors of covariance matrix
 M = (A-np.mean(A.T,axis=1)).T # subtract the mean (along columns)
 [latent,coeff] = np.linalg.eig(np.cov(M)) # attention:not always sorted
 score = np.dot(coeff.T,M) # projection of the data in the new space
 return coeff,score,latent
