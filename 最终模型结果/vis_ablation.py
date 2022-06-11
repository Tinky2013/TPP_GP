
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():

    split = int(0.7 * len(y))

    y_train = y[:split]
    y_test = y[split:]
    print(len(y_train),len(y_test))
    t = np.arange(len(y_train) + len(y_test))

    pred_train = pd.read_csv('ablation/train_result_e7'+df_file+'.csv')
    pred_test = pd.read_csv('ablation/test_result_e7'+df_file+'.csv')

    pred_trainA = pd.read_csv('ablation/train_result_e2' + df_file + '.csv')
    pred_testA = pd.read_csv('ablation/test_result_e2' + df_file + '.csv')

    pred_trainB = pd.read_csv('ablation/train_result_e27' + df_file + '.csv')
    pred_testB = pd.read_csv('ablation/test_result_e27' + df_file + '.csv')

    pred_trainC = pd.read_csv('ablation/train_result_e6' + df_file + '.csv')
    pred_testC = pd.read_csv('ablation/test_result_e6' + df_file + '.csv')

    F = len(t)
    ts = len(y_train)   # test index start
    median = np.zeros(F)
    medianA = np.zeros(F)
    medianB = np.zeros(F)
    medianC = np.zeros(F)

    for i in range(len(y_train)):
        median[i] = np.percentile(pred_train.iloc[i,:], 50)
        medianA[i] = np.percentile(pred_trainA.iloc[i, :], 50)
        medianB[i] = np.percentile(pred_trainB.iloc[i, :], 50)
        medianC[i] = np.percentile(pred_trainC.iloc[i, :], 50)


    for i in range(len(y_test)):
        median[ts + i] = np.percentile(pred_test.iloc[i,:], 50)
        medianA[ts + i] = np.percentile(pred_testA.iloc[i, :], 50)
        medianB[ts + i] = np.percentile(pred_testB.iloc[i, :], 50)
        medianC[ts + i] = np.percentile(pred_testC.iloc[i, :], 50)


    xticks = np.arange(F)
    plt.figure(figsize=(16,5))
    plt.plot(xticks, y, 'black', lw=2, label='observed')
    plt.plot(xticks, median, 'purple', linestyle='dashed', lw=2, label='predict')

    plt.plot(xticks, medianA, 'blue', linestyle='dashed', lw=2, label='mute Long SE')
    plt.plot(xticks, medianB, 'orange', linestyle='dashed', lw=2, label='mute Short SE')
    plt.plot(xticks, medianC, 'cyan', linestyle='dashed', lw=2, label='mute per')

    plt.vlines(x=len(y_train), ymin=0, ymax=ylim, colors="black", linestyles="dashed")
    plt.grid()
    plt.legend()
    plt.ylim((0,ylim))
    plt.title('Data3')
    plt.xlabel('Time (in hours)')
    plt.savefig('d3_ablation.jpg',dpi=600)
    plt.show()


# 实验数据名
ylim = 30
df_file = 'd7'

df = pd.read_csv("data/data7.csv")
y = df['click']

'''
for ablation:
A-mute Long SE
B-mute Short SE
C-mute Per
'''

if __name__ == '__main__':
    main()

