import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano.tensor as tt
import time

# 读取数据
df = pd.read_csv("data/click_count_hour.csv")
y = df['click'][:150]

# 构造训练和测试集
split = int(0.8 * len(y))
y_train = y[:split]
y_test = y[split:]
timeIdx = np.arange(len(y_train) + len(y_test))[:, None]

start = time.time()
def my_callback(trace, draw):
    if len(trace) % 10 == 0:
        end = time.time()
        print("Sample trace: ", len(trace), "Accumulate Running time:", end - start)

def decayKernel(dt):
    '''
    传入一列数据
    '''
    disMat = np.array([[i-j for j in range(len(dt))] for i in range(len(dt))])   # distance matrix
    disMat = np.exp(-disMat)
    disMat[disMat>1] = 0
    return disMat

with pm.Model() as model:
    # Gaussian Process Prior
    mean_f1 = pm.gp.mean.Constant(c=0)
    a = pm.HalfNormal('amplitude', sigma=2)
    l = pm.TruncatedNormal('time-scale', mu=20, sigma=5, lower=0)
    cov_f1 = a ** 2 * pm.gp.cov.ExpQuad(input_dim=1, ls=l)
    GP1 = pm.gp.Latent(mean_func=mean_f1, cov_func=cov_f1)

    mean_f3 = pm.gp.mean.Constant(c=0)
    l3 = pm.TruncatedNormal('time-scale2', mu=20, sigma=5, lower=0)
    cov_f3 = pm.gp.cov.Periodic(input_dim=1, period=24, ls=l3)
    GP3 = pm.gp.Latent(mean_func=mean_f3, cov_func=cov_f3)

    GP = GP1 + GP3
    #GP = GP1
    #GP = GP3
    f = GP.prior('f', X=timeIdx)
    # Decay Kernel
    alpha = pm.HalfNormal('amplification', sigma=10)
    # alpha = 0

    # mean
    Beta = pm.HalfNormal('population_mean', sigma=1)
    # Beta = 0

    Lambda = np.mean(y_train) * Beta + tt.exp(f[:split]) + np.dot(decayKernel(y_train),y_train) * alpha
    pm.Poisson('y_val', mu=Lambda, observed=y_train)
    trace = pm.sample(draws=400, tune=400, chains=1, target_accept=.9, random_seed=42, callback=my_callback)

pm.plot_trace(trace)

print("amplitude:", pm.summary(trace['amplitude']))
print("time-scale:", pm.summary(trace['time-scale']))
print("time-scale2:", pm.summary(trace['time-scale2']))
print("amplification:", pm.summary(trace['amplification']))
print("population_mean:", pm.summary(trace['population_mean']))

with model:
    val_samples = pm.sample_posterior_predictive(trace, random_seed=42)

forecasts_for_train = val_samples['y_val']

F = len(y_train)
low = np.zeros(F)
high = np.zeros(F)
mean = np.zeros(F)
median = np.zeros(F)

for i in range(F):
    low[i] = np.percentile(forecasts_for_train[:, i], 2.5)
    high[i] = np.percentile(forecasts_for_train[:, i], 97.5)
    median[i] = np.percentile(forecasts_for_train[:, i], 50)
    mean[i] = np.mean(forecasts_for_train[:, i])

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
plt.show()


with model:
    Lambda = np.mean(y_test) * Beta + tt.exp(f[split:]) + np.dot(decayKernel(y_test),y_test) * alpha
    y_pred = pm.Poisson('y_pred', mu=Lambda, observed=y_test)
    test_samples = pm.sample_posterior_predictive(trace, var_names=['y_pred'], random_seed=42)

forecasts_for_test = test_samples['y_pred']

Fs = len(y_test)
lows = np.zeros(Fs)
highs = np.zeros(Fs)
means = np.zeros(Fs)
medians = np.zeros(Fs)

for i in range(Fs):
    lows[i] = np.percentile(forecasts_for_test[:, i], 2.5)
    highs[i] = np.percentile(forecasts_for_test[:, i], 97.5)
    medians[i] = np.percentile(forecasts_for_test[:, i], 50)
    means[i] = np.mean(forecasts_for_test[:, i])

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
plt.show()

def MSE(y_true, y_pred):
    return (np.square(y_true-y_pred).sum())/len(y_true)

print("training MSE:", MSE(y_train, mean))
print("testing MSE:", MSE(y_test, means))

