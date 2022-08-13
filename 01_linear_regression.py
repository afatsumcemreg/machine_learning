# importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

pd.set_option('display.float_format', lambda x: '%.2f' %x)

# importing datasets
df = pd.read_csv('C:/Users/test/Desktop/PROGRAMMING/DATASETS/Advertising.csv')
df.columns = [col.lower() for col in df.columns]
df.head()

# selecting dependent and independent variables
y = df[['sales']]
X = df.drop('sales', axis=1)

# setting model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train.shape
X_test.shape
linear_model = LinearRegression().fit(X_train, y_train)

# intercept of linear model
linear_model.intercept_

# coefficients of linear model
linear_model.coef_

# prediction using new data
new_data = [[30], [10], [40]]
new_data = pd.DataFrame(new_data).T
linear_model.predict(new_data)

# rmse train
y_pred = linear_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# r2 train
linear_model.score(X_train, y_train)

# rmse test
y_pred = linear_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# r2 test
linear_model.score(X_test, y_test)

# 10-fold cross validation
np.mean(np.sqrt(-cross_val_score(linear_model, X, y, cv=10, scoring='neg_mean_squared_error')))


# Linear regression with gradient descent
## defining cost function
def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2
    mse = sse / m
    return mse


## defining update rule
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w


## defining train function
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w, cost_function(Y, initial_b, initial_w, X)))
    b = initial_b
    w = initial_w
    cost_history = []
    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)
        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))
    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w


## defining dataset
df = pd.read_csv('C:/Users/test/Desktop/PROGRAMMING/DATASETS/Advertising.csv')
df.columns = [col.lower() for col in df.columns]
X = df['radio']
Y = df['sales']

## hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 10000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)


