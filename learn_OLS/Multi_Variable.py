import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("housing_data.csv")


def prints(df):
    print(list(df.columns))
    print(df.isnull().sum())


def data_cleaning(df):
    df.square_feet = df.square_feet.fillna(df.square_feet.mean())
    df = df.dropna(subset="price").copy()
    df.garage = df.garage.fillna(df.garage.mode()[0])
    df.year_built = df.year_built.fillna(df.year_built.mean())
    df = pd.get_dummies(df, columns=["neighborhood"], drop_first=True)
    df = pd.get_dummies(df, columns=["garage"], drop_first=True)
    df = pd.get_dummies(df, columns=["condition"], drop_first=True)

    return df


def split_data_set(df):
    X_train = df.drop(columns=["price"])
    y_train = df["price"]

    return X_train, y_train


def predict_single_loop(x, w, b):
    """
    Single predict

    Args:
        x (ndarray):
        w (ndarray):
        b (scalar):

    Returns:
        p (scalar): Prediction
    """
    n = x.shape[0]
    p = 0
    for i in range(n):
        p_i = x[i] * w[i]
        p = p + p_i
    p = p + b
    return p


def predict(x, w, b):
    # Takes the dot between x vector and w vector
    p = np.dot(x, w) + b
    return p


def compute_cost(X, y, w, b):
    """
    Compute cost
    Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m, )): Target values
        w (ndarray (n, )): model parameters
        b (scalar)       : model parameter

    Returns:
        cost (scalar): cost

    """

    m = X.shape[0]  # Observations
    cost = 0.0
    for i in range(m):  # loop through observations
        f_wb_i = np.dot(X[i], w) + b  # compute DOT between all x values and w values
        cost = cost + (f_wb_i - y[i]) ** 2
    cost = cost / (2 * m)
    return cost


def compute_gradients(X, y, w, b):
    """
    Computes gradients
    """

    m, n = X.shape

    dj_dw = np.zeros((n,))
    dj_db = 0
    # compute gradients

    for i in range(m):  # loop through every observation
        f_wb_i = np.dot(X[i], w) + b
        error = f_wb_i - y[i]  # compute error
        for j in range(n):  # for each feature
            dj_dw[j] = dj_dw[j] + error * X[i, j]  # compute gradient
        dj_db = dj_db + error  # outside of loop calc b
    dj_dw = dj_dw / m  # average gradient
    dj_db = dj_db / m  # average gradient

    return dj_dw, dj_db


def gradient_descent(
    X, y, w_in, b_in, cost_function, compute_gradients, alpha, num_iters
):
    """
    Batch gradient descent
    """

    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        # compute gradient
        dj_dw, dj_db = compute_gradients(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_history.append(cost_function(X, y, w, b))
            if i % math.ceil(num_iters / 10) == 0:
                print(f"iteration {i}: Cost {J_history[-1]:8.2f}")

    return w, b, J_history


if __name__ == "__main__":
    cleaned_data = data_cleaning(data)
    # prints(cleaned_data)
    X_train, y_train = split_data_set(cleaned_data)
    X_train = X_train.values
    y_train = y_train.values
    scalerX = StandardScaler()
    scalerY = StandardScaler()
    X_train_norm = scalerX.fit_transform(X_train)
    # y_train_norm = scalerY.fit_transform(y_train.reshape(-1, 1)).ravel()
    n = X_train.shape[1]
    b_init = 0
    w_init = np.zeros(n)
    alpha = 0.001
    iterations = 1000

    w_final, b_final, J_hist = gradient_descent(
        X_train_norm,
        y_train,
        w_init,
        b_init,
        compute_cost,
        compute_gradients,
        alpha,
        iterations,
    )

    print(f"w: {w_final}, b: {b_final}")

    plt.figure(figsize=(12, 4))
    plt.plot(J_hist)
    plt.title("cost vs iteration")
    plt.ylabel("cost")
    plt.xlabel("iterations")
    plt.show()

    plt.figure(figsize=(12, 4))
