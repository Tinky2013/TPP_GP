
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARMA
import statsmodels.api as sm

def MSE(y_true, y_pred):
    return (np.square(y_true - y_pred).sum()) / len(y_true)

def MAE(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / len(y_true)

def main():
    trainMSE, testMSE, trainMAE, testMAE = [], [], [], []
    for trajectory in ['0','1','2','3','4','5','6','7','8','9']:
        df = pd.read_csv("syn_data/"+PATH+".csv")
        y = df[trajectory]
        split = int(0.667 * len(y))

        y_train = y[:split]
        y_test = y[split:]

        order = (3,0,3)
        # 季节自回归阶数，季节差分阶数，季节移动平均阶数，单个季节时间步数
        seasonal_order = (1, 0, 1, 7)
        tempModel = sm.tsa.statespace.SARIMAX(y_train,order=order,seasonal_order=seasonal_order).fit()

        delta = tempModel.fittedvalues - y_train # 残差
        score = 1 - delta.var()/y_train.var()

        # 这里传入的是start,end，会获取index[start], index[end]
        y_train_pred = tempModel.predict(0, 119)
        y_test_pred = tempModel.predict(120, 179)

        trainMSE.append(MSE(y_train, y_train_pred))
        testMSE.append(MSE(y_test, y_test_pred))
        trainMAE.append(MAE(y_train, y_train_pred))
        testMAE.append(MAE(y_test, y_test_pred))

    result = {
        'trainMSE': trainMSE,
        'testMSE': testMSE,
        'trainMAE': trainMAE,
        'testMAE': testMAE,
    }
    result = pd.DataFrame(result)
    result.to_csv('result_sarimax_' + PATH + '.csv', index=False)

PATH = 'double3'

if __name__ == '__main__':
    main()
