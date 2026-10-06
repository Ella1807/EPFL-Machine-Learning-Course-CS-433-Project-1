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

#y target values, tx feature matrix, initial_w starting weights
#max_iters #of gradient descent iterations, gamma stepsize/learning rate
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
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