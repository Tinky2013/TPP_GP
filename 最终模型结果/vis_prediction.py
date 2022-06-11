
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():

    split = int(0.7 * len(y))

    y_train = y[:split]
    y_test = y[split:]
    print(len(y_train),len(y_test))
    t = np.arange(len(y_train) + len(y_test))

    forecasts_for_train = pd.read_csv('ablation/train_result_'+save_path+'.csv')
    forecasts_for_test = pd.read_csv('ablation/test_result_'+save_path+'.csv')

    F = len(t)
    ts = len(y_train)   # test index start
    low = np.zeros(F)
    high = np.zeros(F)
    mean = np.zeros(F)
    median = np.zeros(F)

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


    xticks = np.arange(F)
    plt.figure(figsize=(16,5))
    plt.fill_between(xticks, low, high, color='purple', alpha=0.2)
    plt.plot(xticks, y, 'black', lw=2, label='observed')
    plt.plot(xticks, median, 'purple', linestyle='dashed', lw=2, label='predict')
    plt.vlines(x=len(y_train), ymin=0, ymax=np.max(high), colors="black", linestyles="dashed")
    plt.grid()
    plt.legend()
    plt.ylim((0,45))
    plt.title('Data1')
    plt.xlabel('Time (in hours)')
    plt.savefig('d1_full.jpg',dpi=600)
    plt.show()

save_path = 'e7d1' # 每个跑实验改这个路径

df = pd.read_csv("data/data1.csv")
y = df['click']

if __name__ == '__main__':
    main()

