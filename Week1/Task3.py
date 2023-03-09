# 练习3
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from scipy.optimize import leastsq



raw_data = pd.read_csv('./kc_house_data.csv')
raw_data.duplicated().sum()
X = raw_data.drop(['id', 'date', 'price'], axis=1)
y = raw_data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1026)

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
y_train = np.array(y_train).reshape((-1, 1))
X_train = np.insert(X_train, 0, 1, axis=1)
print(X_train)




# print(X_train.shape, X_train.T.shape)
# print(y_train.shape)
# print(type(X_train))

# 需要让训练集X_train的列向量为特征向量,在这里转置一次,转置前行向量为特征向量
# 公式: W的转置=(X@X转置)**-1@X@Y转置

# X_train = X_train.T
# y_train = y_train.T
# print(X_train.shape, y_train.shape)
# W_tr = (np.linalg.inv(X_train@X_train.T))@X_train@y_train.T
# W = W_tr.T
# a = W@X_train

# np.set_printoptions(threshold=np.inf)
# print(y_train)



# 梯度下降算法
def loss_func(x, y, w):
    k = y.reshape((-1, 1)) - x@w
    k = k.T@k/(2*len(k))
    print("loss:", k)


def gradient(x, y, w):
    m, n = x.shape
    res = np.zeros((n, 1))
    for i in range(m):
        p = y[i]-x[i].T@w
        for j in range(n):
            res[j] -= p*x[i][j]
    return res


def gradient1(x, y, w):
    m, n = x.shape
    s1 = x@w-y
    s2 = (x.T@s1)/m
    return s2

# def ridge_gradient(x, y, w, lam):
#     m, n = x.shape
#     res = np.zeros((n, 1))
#     for i in range(m):


W = np.zeros((19, 1))
# b = np.zeros(((15129, 1)))
# W = np.random.random((18, 1))
iters = 100000
lr = 0.2
lamda = 0.1


for i in range(iters):

    # y_h = (W.T@X_train.T).T
    # print(X_train.shape)
    # print(y_train.shape)
    loss_func(X_train, y_train, W)  # 15129,18   15129,1
    dw = gradient1(X_train, y_train, W)
    W = W - lr*dw


y_pre = X_train@W
print(y_pre)


# model_slect = GradientBoostingRegressor()
# parameters = {'loss': ['ls', 'lad', 'huber', 'quantile'], 'learning_rate':[0.1, 0.2], 'min_samples_leaf':[1, 2, 3, 4]}
# time_start = time.time()
# model_gs = GridSearchCV(estimator=model_slect, param_grid=parameters, verbose=3)
# model_gs.fit(X, y)

# model = LinearRegression()
# model.fit(X_train, y_train)
# print(model.predict(X_test))
# print(y_test)

