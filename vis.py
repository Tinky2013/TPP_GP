import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano.tensor as tt
import time

from data_loader import DataLoader

dataLoader = DataLoader()


y = dataLoader.y
split = dataLoader.split
y_train = y[:split]
y_test = y[split:]
timeIdx = dataLoader.timeIdx
t = np.arange(len(y_train) + len(y_test))
t_train = t[:split]
t_test = t[split:]

forecasts_for_train = pd.read_csv('train_result_genpoi.csv')

# 一行为一个样本点
forecasts_for_test = pd.read_csv('test_result_genpoi.csv')

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




Fs = len(y_test)
lows = np.zeros(Fs)
highs = np.zeros(Fs)
means = np.zeros(Fs)
medians = np.zeros(Fs)

for i in range(Fs):
    lows[i] = np.percentile(forecasts_for_test.iloc[i,], 2.5)
    highs[i] = np.percentile(forecasts_for_test.iloc[i,], 97.5)
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

def MSE(y_true, y_pred):
    return (np.square(y_true-y_pred).sum())/len(y_true)

print("training MSE:", MSE(y_train, mean))
print("testing MSE:", MSE(y_test, means))

