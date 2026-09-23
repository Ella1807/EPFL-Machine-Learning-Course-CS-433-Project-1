import numpy as np

#y target values, tx feature matrix, initial_w starting weights
#max_iters #of gradient descent iterations, gamma stepsize/learning rate
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    w = initial_w.copy()

    for _ in range(max_iters):
        error = y - tx @ w
        gradient = -(tx.T @ error) / len(y)
        w = w - gamma * gradient #grad descent updates

    error = y - tx @ w #final loss
    loss = 0.5 * np.mean(error ** 2) #formule cours

    return w, loss