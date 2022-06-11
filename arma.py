
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARMA


df = pd.read_csv("syn_data/trend6.csv")
y = df['9']
split = int(0.667 * len(y))

y_train = y[:split]
y_test = y[split:]

timeIdx = np.arange(len(y_train) + len(y_test))[:, None]
t = np.arange(len(y_train) + len(y_test))

t_train = t[:split]
t_test = t[split:]

order = (3,3)
tempModel = ARMA(y_train,order).fit()

delta = tempModel.fittedvalues - y_train # 残差
score = 1 - delta.var()/y_train.var()

# 这里传入的是start,end，会获取index[start], index[end]
y_train_pred = tempModel.predict(0, 119, dynamic=True)
y_test_pred = tempModel.predict(120, 179, dynamic=True)

train_result_dt = pd.DataFrame(y_train_pred)
train_result_dt.to_csv("train_result_arma33_d3.csv",index=False)
test_result_dt = pd.DataFrame(y_test_pred)
test_result_dt.to_csv("test_result_arma33_d3.csv",index=False)

def MSE(y_true, y_pred):
    return (np.square(y_true-y_pred).sum())/len(y_true)

def MAE(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / len(y_true)

print("training MSE:", MSE(y_train, y_train_pred), "testing MSE:", MSE(y_test, y_test_pred))
print("training MAE:", MAE(y_train, y_train_pred), "testing MAE:", MAE(y_test, y_test_pred))

