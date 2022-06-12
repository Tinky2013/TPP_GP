
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():

    split = int(0.7 * len(y))

    y_train = y[:split]
    y_test = y[split:]
    print(len(y_train),len(y_test))
    t = np.arange(len(y_train) + len(y_test))

    forecasts_for_train = pd.read_csv('2per/train_result_e7d2.csv')
    forecasts_for_test = pd.read_csv('2per/test_result_e7d2.csv')

    forecasts_for_train1 = pd.read_csv('2per/train_result_e8d2.csv')
    forecasts_for_test1 = pd.read_csv('2per/test_result_e8d2.csv')

    F = len(t)
    ts = len(y_train)   # test index start
    low = np.zeros(F)
    high = np.zeros(F)
    mean = np.zeros(F)
    median = np.zeros(F)
    low1 = np.zeros(F)
    high1 = np.zeros(F)
    mean1 = np.zeros(F)
    median1 = np.zeros(F)

    for i in range(len(y_train)):
        low[i] = np.percentile(forecasts_for_train.iloc[i,:], 2.5)
        high[i] = np.percentile(forecasts_for_train.iloc[i,:], 97.5)
        median[i] = np.percentile(forecasts_for_train.iloc[i,:], 50)
        mean[i] = np.mean(forecasts_for_train.iloc[i,:])

    for i in range(len(y_test)):
        low[ts+i] = np.percentile(forecasts_for_test.iloc[i,:], 5)
        high[ts+i] = np.percentile(forecasts_for_test.iloc[i,:], 95)
        median[ts+i] = np.percentile(forecasts_for_test.iloc[i,:], 50)
        mean[ts+i] = np.mean(forecasts_for_test.iloc[i,:])

    for i in range(len(y_train)):
        low1[i] = np.percentile(forecasts_for_train1.iloc[i,:], 2.5)
        high1[i] = np.percentile(forecasts_for_train1.iloc[i,:], 97.5)
        median1[i] = np.percentile(forecasts_for_train1.iloc[i,:], 50)
        mean1[i] = np.mean(forecasts_for_train1.iloc[i,:])

    for i in range(len(y_test)):
        low1[ts+i] = np.percentile(forecasts_for_test1.iloc[i,:], 5)
        high1[ts+i] = np.percentile(forecasts_for_test1.iloc[i,:], 95)
        median1[ts+i] = np.percentile(forecasts_for_test1.iloc[i,:], 50)
        mean1[ts+i] = np.mean(forecasts_for_test1.iloc[i,:])


    xticks = np.arange(F)
    plt.figure(figsize=(16,5))
    #plt.fill_between(xticks, low, high, color='purple', alpha=0.2)
    #plt.fill_between(xticks, low1, high1, color='orange', alpha=0.2)
    plt.plot(xticks, y, 'black', lw=2, label='observed')
    plt.plot(xticks, median, 'purple', linestyle='dashed', lw=2, label='Basic Model')
    plt.plot(xticks, median1, 'orange', linestyle='dashed', lw=2, label='DP Model')
    plt.vlines(x=len(y_train), ymin=0, ymax=np.max(high), colors="black", linestyles="dashed")
    plt.grid()
    plt.legend()
    plt.ylim((0,15))
    plt.title('Data4')
    plt.xlabel('Time (in hours)')
    plt.savefig('d2_2per.jpg',dpi=600)
    plt.show()


df = pd.read_csv("data/data2.csv")
y = df['click']

if __name__ == '__main__':
    main()

