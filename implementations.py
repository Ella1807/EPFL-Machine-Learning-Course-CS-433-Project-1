import numpy as np

#y target values, tx feature matrix, initial_w starting weights
#max_iters #of gradient descent iterations, gamma stepsize/learning rate
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent (MSE loss).

    Args:
        y: target values, shape (N,)
        tx: feature matrix, shape (N, D)
        initial_w: initial weight vector, shape (D,)
        max_iters: number of gradient descent iterations
        gamma: step size (learning rate)

    Returns:
        w: last weight vector, shape (D,)
        loss: MSE loss (with the 1/2 factor) of the last w
    """
    w = initial_w.copy()

    for _ in range(max_iters):
        error = y - tx @ w
        gradient = -(tx.T @ error) / len(y)
        w = w - gamma * gradient #grad descent updates

    error = y - tx @ w #final loss
    loss = 0.5 * np.mean(error ** 2) #formule cours

    return w, loss

#y target values, tx feature matrix, initial_w starting weights
#max_iters #of gradient descent iterations, gamma stepsize/learning rate
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (batch size 1).

    Args:
        y: target values, shape (N,)
        tx: feature matrix, shape (N, D)
        initial_w: initial weight vector, shape (D,)
        max_iters: number of SGD iterations
        gamma: step size (learning rate)

    Returns:
        w: last weight vector, shape (D,)
        loss: MSE loss (with the 1/2 factor) of the last w,
            computed on the whole dataset
    """
    
    w = initial_w.copy()

    for _ in range(max_iters):

        # choose one random datapoint
        i = np.random.randint(len(y))

        x_i = tx[i]
        y_i = y[i]

        # error for this datapoint
        error = y_i - x_i @ w

        # stochastic gradient
        gradient = -x_i * error

        # gradient descent update
        w = w - gamma * gradient

    # final loss on the COMPLETE dataset
    error = y - tx @ w
    loss = 0.5 * np.mean(error ** 2)

    return w, loss

#y target values, tx feature matrix, lambda_ regularization parameter
def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations.

    Args:
        y: target values, shape (N,)
        tx: feature matrix, shape (N, D)
        lambda_: regularization parameter

    Returns:
        w: optimal weights, shape (D,)
        loss: MSE loss (without the penalty term)
    """
    N, D = tx.shape

    # normal equations of ridge: (X^T X + 2*N*lambda I) w = X^T y
    A = tx.T @ tx + 2 * N * lambda_ * np.eye(D)
    b = tx.T @ y
    w = np.linalg.solve(A, b) #solve plutôt que inverse, plus stable

    # loss WITHOUT the penalty term (consigne du projet)
    error = y - tx @ w
    loss = 0.5 * np.mean(error ** 2)

    return w, loss