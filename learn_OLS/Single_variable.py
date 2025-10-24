import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

housing = pd.read_csv("housing_data.csv")

housing.square_feet = housing["square_feet"].fillna(housing["square_feet"].mean())
housing.year_built = housing.year_built.fillna(housing.year_built.mean())
housing.garage = housing.garage.fillna(housing.garage.mode()[0])
housing = housing.dropna(subset=["price"])

# Training set
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_train = scaler_x.fit_transform(housing.square_feet.values.reshape(-1, 1))
# Real y values
y_train = scaler_y.fit_transform(housing["price"].values.reshape(-1, 1))


def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        total_cost += cost
    total_cost = (1 / (2 * m)) * total_cost

    return total_cost


def compute_gradients(x, y, w, b):
    dj_dw = 0
    dj_db = 0
    m = x.shape[0]

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(
    x, y, w_in, b_in, alpha, iterations, compute_cost, compute_gradients
):
    J_history = []
    p_history = []

    w = w_in
    b = b_in

    # we want to update the gradients simulatenously

    for i in range(iterations):
        dj_dw, dj_db = compute_gradients(x, y, w, b)

        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        if i < 100000:
            J_history.append(compute_cost(x, y, w, b))
            p_history.append([w, b])

        if i % 100 == 0:
            print(J_history[-1])

    return w, b, J_history, p_history


w, b, J_history, p_history = gradient_descent(
    x_train, y_train, 5, 5, 0.01, 1000, compute_cost, compute_gradients
)

print(f"W: {w}, b: {b}")

y_hat_norm = w * x_train.ravel() + b
y_hat_dollars = scaler_y.inverse_transform(y_hat_norm.reshape(-1, 1)).ravel()

plt.figure(figsize=(14, 6))
sns.scatterplot(data=housing, x="square_feet", y="price")
sns.lineplot(x=housing.square_feet, y=y_hat_dollars)
plt.show()

# plot cost versus iterations
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_history[:100])
ax2.plot(500 + np.arange(len(J_history[500:])), J_history[500:])
ax1.set_title("Cost vs. iterations (start)")
ax2.set_title("Cost vs iterations (end)")
plt.show()
