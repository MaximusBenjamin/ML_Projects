import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("housing_data.csv")

print(list(data.columns))

# Clean data BEFORE splitting to ensure X and y have matching shapes
data_clean = data[["square_feet", "bedrooms", "price"]].dropna()
data_clean.reset_index(drop=True, inplace=True)

X_train = data_clean[["square_feet", "bedrooms"]]
y_train = data_clean["price"]

x_scaler = StandardScaler()
X_train_norm = x_scaler.fit_transform(X_train)
X_features = ["square_feet", "bedrooms"]

sgdr = SGDRegressor(loss="squared_error", max_iter=1000, alpha=0.1)
sgdr.fit(X_train_norm, y_train)
print(sgdr)
print(
    f"Number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}"
)

b_norm = sgdr.intercept_
w_norm = sgdr.coef_
