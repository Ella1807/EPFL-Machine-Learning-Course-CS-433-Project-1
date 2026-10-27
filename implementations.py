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

#y target values, tx feature matrix
def least_squares(y, tx):
    """Linear regression using the least-squares solution.

    Args:
        y: target values, shape (N,)
        tx: feature matrix, shape (N, D)

    Returns:
        w: optimal weight vector, shape (D,)
        loss: MSE loss (with the 1/2 factor)
    """
    A = tx.T @ tx
    b = tx.T @ y

    # Solve the normal equations: (X^T X)w = X^T y
    w = np.linalg.solve(A, b)

    # Compute the loss at the optimal weights
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

#helper function
def sigmoid(t):
    """Compute the sigmoid function."""
    return 1.0 / (1.0 + np.exp(-t))


def calculate_logistic_loss(y, tx, w):
    """Compute the average negative log-likelihood.

    Args:
        y: Target values, shape (N,), containing 0 or 1.
        tx: Feature matrix, shape (N, D).
        w: Weight vector, shape (D,).

    Returns:
        Scalar loss.
    """
    scores = tx @ w

    # Numerically stable equivalent of:
    # -mean(y * log(sigmoid(scores))
    #       + (1-y) * log(1-sigmoid(scores)))
    loss = np.mean(np.logaddexp(0, scores) - y * scores)

    return loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent.

    Args:
        y: Target values, shape (N,), containing 0 or 1.
        tx: Feature matrix, shape (N, D).
        initial_w: Initial weight vector, shape (D,).
        max_iters: Number of gradient descent iterations.
        gamma: Learning rate.

    Returns:
        w: Final weight vector, shape (D,).
        loss: Final average negative log-likelihood.
    """
    w = initial_w.copy()
    N = len(y)

    for _ in range(max_iters):
        # Predicted probabilities
        predictions = sigmoid(tx @ w)

        # Gradient of the logistic loss
        gradient = tx.T @ (predictions - y) / N

        # Gradient descent update
        w = w - gamma * gradient

    # Evaluate loss using the final weights
    loss = calculate_logistic_loss(y, tx, w)

    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.

    Args:
        y: Target values, shape (N,), containing 0 or 1.
        tx: Feature matrix, shape (N, D).
        lambda_: Regularization parameter.
        initial_w: Initial weight vector, shape (D,).
        max_iters: Number of gradient descent iterations.
        gamma: Learning rate.

    Returns:
        w: Final weight vector, shape (D,).
        loss: Final average negative log-likelihood,
              excluding the regularization penalty.
    """
    w = initial_w.copy()
    N = len(y)

    for _ in range(max_iters):
        predictions = sigmoid(tx @ w)

        # Gradient of logistic loss + lambda * ||w||^2
        gradient = tx.T @ (predictions - y) / N + 2 * lambda_ * w

        w = w - gamma * gradient

    # Exclude the regularization penalty from the returned loss
    loss = calculate_logistic_loss(y, tx, w)

    return w, loss
