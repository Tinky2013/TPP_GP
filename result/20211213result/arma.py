
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARMA



from data_loader import DataLoader

df = pd.read_csv("data/data7.csv")
y = df['click']
split = int(0.7 * len(y))

y_train = y[:split]
y_test = y[split:]

timeIdx = np.arange(len(y_train) + len(y_test))[:, None]
t = np.arange(len(y_train) + len(y_test))

t_train = t[:split]
t_test = t[split:]
# lag_acf = acf(y_train, nlags=48)
# lag_pacf = pacf(y_train, nlags=48, method='ols')
#
# plt.plot(lag_acf)
# plt.show()
# plt.plot(lag_pacf)
# plt.show()


order = (3,3)
tempModel = ARMA(y_train,order).fit()

delta = tempModel.fittedvalues - y_train # 残差
score = 1 - delta.var()/y_train.var()

# 这里传入的是start,end，会获取index[start], index[end]
y_train_pred = tempModel.predict(0, 356)
y_test_pred = tempModel.predict(357, 509)

train_result_dt = pd.DataFrame(y_train_pred)
train_result_dt.to_csv("train_result_arma33_d3.csv",index=False)
test_result_dt = pd.DataFrame(y_test_pred)
test_result_dt.to_csv("test_result_arma33_d3.csv",index=False)

def MSE(y_true, y_pred):
    print(len(y_true), len(y_pred))
    return (np.square(y_true-y_pred).sum())/len(y_true)

print("training MSE:", MSE(y_train, y_train_pred))
print("testing MSE:", MSE(y_test, y_test_pred))

