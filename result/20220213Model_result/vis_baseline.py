
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():

    split = int(0.7 * len(y))

    y_train = y[:split]
    y_test = y[split:]
    print(len(y_train),len(y_test))
    t = np.arange(len(y_train) + len(y_test))

    pred_train = pd.read_csv('baseline/train_result_e7'+df_file+'.csv')
    pred_test = pd.read_csv('baseline/test_result_e7'+df_file+'.csv')

    pred_trainA = pd.read_csv('baseline/train_result_arma33_' + df_file + '.csv')
    pred_testA = pd.read_csv('baseline/test_result_arma33_' + df_file + '.csv')

    pred_trainB = pd.read_csv('baseline/train_result_prophet_' + df_file + '.csv')
    pred_testB = pd.read_csv('baseline/test_result_prophet_' + df_file + '.csv')


    F = len(t)
    ts = len(y_train)   # test index start
    median = np.zeros(F)

    for i in range(len(y_train)):
        median[i] = np.percentile(pred_train.iloc[i,:], 50)


    for i in range(len(y_test)):
        median[ts + i] = np.percentile(pred_test.iloc[i,:], 50)


    xticks = np.arange(F)
    plt.figure(figsize=(16,5))
    plt.plot(xticks, y, 'black', lw=2, label='observed')
    plt.plot(xticks, median, 'purple', linestyle='dashed', lw=2, label='ours')

    plt.plot(xticks, np.array(pd.concat([pred_trainA,pred_testA])).flatten(), 'blue', linestyle='dashed', lw=2, label='ARMA')
    plt.plot(xticks, np.array(pd.concat([pred_trainB,pred_testB])).flatten(), 'orange', linestyle='dashed', lw=2, label='Prophet')

    plt.vlines(x=len(y_train), ymin=0, ymax=ylim, colors="black", linestyles="dashed")
    plt.grid()
    plt.legend()
    plt.ylim((0,ylim))
    plt.title('Data3')
    plt.xlabel('Time (in hours)')
    #plt.savefig('d3_ablation.jpg',dpi=600)
    plt.show()

    median=pd.DataFrame(median)
    median.to_csv('median_data1.csv',index=False)

# 实验数据名
ylim = 30
df_file = 'd1'

df = pd.read_csv("data/data1.csv")
y = df['click']

'''
for baseline:
A-ARMA
B-Prophet
'''

if __name__ == '__main__':
    main()

