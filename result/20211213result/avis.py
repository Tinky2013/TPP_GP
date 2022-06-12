
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from data_loader import DataLoader




def MSE(y_true, y_pred):
    return (np.square(y_true - y_pred).sum()) / len(y_true)

def MAE(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / len(y_true)

def main():

    split = int(0.7 * len(y))

    y_train = y[:split]
    y_test = y[split:]
    print(len(y_train),len(y_test))
    timeIdx = np.arange(len(y_train) + len(y_test))[:, None]
    t = np.arange(len(y_train) + len(y_test))

    t_train = t[:split]
    t_test = t[split:]

    forecasts_for_train = pd.read_csv('train_result_'+save_path+'.csv')

    F = len(y_train)
    low = np.zeros(F)
    high = np.zeros(F)
    mean = np.zeros(F)
    median = np.zeros(F)

    for i in range(F):
        low[i] = np.percentile(forecasts_for_train.iloc[i,:], 2.5)
        high[i] = np.percentile(forecasts_for_train.iloc[i,:], 97.5)
        median[i] = np.percentile(forecasts_for_train.iloc[i,:], 50)
        mean[i] = np.mean(forecasts_for_train.iloc[i,:])

    xticks = np.arange(F)
    plt.figure(figsize=(8,6))
    plt.errorbar(xticks, median,
                 yerr=[median-low, high-median],
                 capsize=2, fmt='.', linewidth=1,
                 label='2.5, 50, 97.5 percentiles')
    plt.plot(xticks, mean, 'x', label='mean')
    plt.plot(xticks, y_train, 's', label='observed')
    plt.legend()
    plt.title('Forecasts for train')
    plt.xlabel('Day')
    #plt.show()


    # 一行为一个样本点
    forecasts_for_test = pd.read_csv('test_result_'+save_path+'.csv')


    Fs = len(y_test)
    lows = np.zeros(Fs)
    highs = np.zeros(Fs)
    means = np.zeros(Fs)
    medians = np.zeros(Fs)

    for i in range(Fs):
        lows[i] = np.percentile(forecasts_for_test.iloc[i,], 5)
        highs[i] = np.percentile(forecasts_for_test.iloc[i,], 95)
        medians[i] = np.percentile(forecasts_for_test.iloc[i,], 50)
        means[i] = np.mean(forecasts_for_test.iloc[i,])

    xticks = np.arange(Fs)
    plt.figure(figsize=(8,6))
    plt.errorbar(xticks, medians,
                 yerr=[medians-lows, highs-medians],
                 capsize=2, fmt='.', linewidth=1,
                 label='2.5, 50, 97.5 percentiles')
    plt.plot(xticks, means, 'x', label='mean')
    plt.plot(xticks, y_test, 's', label='observed')
    plt.legend()
    plt.title('Forecasts for test')
    plt.xlabel('Day')
    #plt.show()

    print(len(y_train), len(median), len(y_test), len(medians))

    print("training MSE:", MSE(y_train, median), "testing MSE:", MSE(y_test, medians))
    print("training MAE:", MAE(y_train, median), "testing MAE:", MAE(y_test, medians))

# 'arma33_dx', 'prophet_dx', 'dfx_hpoi', 'e2dx', 'e27dx'
save_path = 'e27d1' # 每个跑实验改这个路径

df = pd.read_csv("data/data1.csv")
y = df['click']

if __name__ == '__main__':
    main()

