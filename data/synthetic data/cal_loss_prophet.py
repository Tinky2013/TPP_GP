
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fbprophet


def MSE(y_true, y_pred):
    return (np.square(y_true - y_pred).sum()) / len(y_true)

def MAE(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / len(y_true)

def main():
    trainMSE, testMSE, trainMAE, testMAE = [], [], [], []
    for trajectory in ['0','1','2','3','4','5','6','7','8','9']:
        m = fbprophet.Prophet()
        #df = pd.read_csv("syn_data/"+PATH+".csv")
        df = pd.read_csv("trend8.csv")
        df1 = pd.read_csv("syn_data/timestump.csv")
        y = df[trajectory]
        t = df1['day']
        split = int(0.667 * len(y))

        y_train = y[:split]
        y_test = y[split:]
        t_train = t[:split]
        t_test = t[split:]

        df = pd.DataFrame([])

        df['ds'] = t_train
        df['y'] = y_train

        m.fit(df)
        future = m.make_future_dataframe(periods=len(t_test))
        forecast = m.predict(future)

        x1 = forecast['ds']
        y1 = forecast['yhat']

        y_train_pred = y1[:split]
        y_test_pred = y1[split:]

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
    result.to_csv('result_prophet_' + PATH + '.csv', index=False)

PATH = 'trend8'

if __name__ == '__main__':
    main()
