import numpy as np


def normal_equation(x,y):
    X_ones = np.c_[np.ones((100, 1)), x.T]
    theta_best = np.linalg.inv(X_ones.T.dot(X_ones)).dot(X_ones.T).dot(y.T)

    X_predict = np.array([[0], [1]])
    X_new_b = np.c_[np.ones((2, 1)), X_predict]

    y_predict = X_new_b.dot(theta_best)
    print("w = "+str(theta_best[1]))
    print("b = "+str(theta_best[0]))

    return X_predict, y_predict
