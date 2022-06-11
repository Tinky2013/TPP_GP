
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARMA
import statsmodels.api as sm


df = pd.read_csv("data/real world data/data2.csv")
y = df['dt']
split = int(0.667 * len(y))

y_train = y[:split]
y_test = y[split:]

timeIdx = np.arange(len(y_train) + len(y_test))[:, None]
t = np.arange(len(y_train) + len(y_test))
t_train = t[:split]
t_test = t[split:]

order = (3,0,3)
# 季节自回归阶数，季节差分阶数，季节移动平均阶数，单个季节时间步数
seasonal_order = (1, 0, 1, 24)
tempModel = sm.tsa.statespace.SARIMAX(y_train,order=order,seasonal_order=seasonal_order).fit()

delta = tempModel.fittedvalues - y_train # 残差
score = 1 - delta.var()/y_train.var()

print(len(y_train)-1)

# 这里传入的是start,end，会获取index[start], index[end]
y_train_pred = tempModel.predict(0, len(y_train)-1)
y_test_pred = tempModel.predict(len(y_train), len(y_train)+len(y_test)-1)

train_result_dt = pd.DataFrame({'y_train': y_train,
                                'y_train_pred': y_train_pred,
                                'y_test': y_test,
                                'y_test_pred': y_test_pred,
                                })
train_result_dt.to_csv("result_sarimax33.csv",index=False)

def MSE(y_true, y_pred):
    return (np.square(y_true-y_pred).sum())/len(y_true)

def MAE(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / len(y_true)

print(len(y_train),len(y_train_pred),len(y_test),len(y_test_pred))
print(y_test)
print(y_test_pred)

print("training MSE:", MSE(y_train, y_train_pred), "testing MSE:", MSE(y_test, y_test_pred))
print("training MAE:", MAE(y_train, y_train_pred), "testing MAE:", MAE(y_test, y_test_pred))

