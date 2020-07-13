import numpy as np


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


def propagate(w, b, x, y):

    m = x.shape[0]

    # FORWARD PROPAGATION (FROM X TO COST)
    cost = 0.5 / m * np.sum(pow(np.dot(w.T, x.T) + b - y, 2))

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = 1 / m * np.dot(x, (np.dot(w.T, x.T) + b - y).T)
    db = 1 / m * np.sum(np.dot(w.T, x.T) + b - y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, x, y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):

        # Cost and gradient calculation
        grads, cost = propagate(w, b, x, y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule
        w = w - dw * learning_rate
        b = b - db * learning_rate

        # Record the costs
        costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


# GRADED FUNCTION: model

def model(x, y, num_iterations=2000, learning_rate=0.5, print_cost=False):

    # initialize parameters with zeros
    w, b = initialize_with_zeros(x.shape[1])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, x, y, num_iterations, learning_rate, print_cost=True)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    d = {"costs": costs,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d