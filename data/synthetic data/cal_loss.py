import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def MSE(y_true, y_pred):
    return (np.square(y_true - y_pred).sum()) / len(y_true)

def MAE(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / len(y_true)

def main():
    trainMSE, testMSE, trainMAE, testMAE = [], [], [], []
    for trajectory in ['0','1','2','3','4','5','6','7','8','9']:

        save_path = PATH+'_'+trajectory  # 每个跑实验改这个路径

        df = pd.read_csv("syn_data/"+PATH+".csv")
        y = df[trajectory]

        # 训练测试集（读取原始数据）
        split = int(0.667 * len(y))
        y_train = y[:split]
        y_test = y[split:]
        # 训练集读取
        forecasts_for_train = pd.read_csv('pymc_result_all/train_result_'+save_path+'.csv')
        # 中位数估计
        median = [np.percentile(forecasts_for_train.iloc[i,:], 50)
                  for i in range(len(y_train))]

        # 测试集读取
        forecasts_for_test = pd.read_csv('pymc_result_all/test_result_'+save_path+'.csv')
        # 中位数估计
        medians = [np.percentile(forecasts_for_test.iloc[i,], 50)
                   for i in range(len(y_test))]

        trainMSE.append(MSE(y_train, median))
        testMSE.append(MSE(y_test, medians))
        trainMAE.append(MAE(y_train, median))
        testMAE.append(MAE(y_test, medians))

    result = {
        'trainMSE': trainMSE,
        'testMSE': testMSE,
        'trainMAE': trainMAE,
        'testMAE': testMAE,
    }
    result = pd.DataFrame(result)
    result.to_csv('result_'+PATH+'.csv',index=False)

PATH = 'trend8'

if __name__ == '__main__':
    main()

