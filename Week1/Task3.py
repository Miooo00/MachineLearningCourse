# 练习3

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

raw_data = pd.read_csv('./kc_house_data.csv')
raw_data.duplicated()

X = raw_data.drop(['id', 'date', 'price'], axis=1)
y = raw_data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1026)

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
