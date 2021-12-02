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

def lambda_Structure():

    return Lambda

with pm.Model() as model:
    # Decay Kernel
    alpha = pm.HalfNormal('amplification', sigma=10)
    # alpha = 0

    # mean
    Beta = pm.HalfNormal('population_mean', sigma=1)
    # Beta = 0

    # Gaussian Process Prior
    mean_f1 = pm.gp.mean.Constant(c=0)
    a = pm.HalfNormal('amplitude', sigma=2)
    l = pm.TruncatedNormal('time-scale', mu=20, sigma=5, lower=0)
    cov_f1 = a ** 2 * pm.gp.cov.ExpQuad(input_dim=1, ls=l)
    GP1 = pm.gp.Latent(mean_func=mean_f1, cov_func=cov_f1)
    #GP1 = pm.gp.Marginal(mean_f1, cov_f1)

    mean_f3 = pm.gp.mean.Constant(c=0)
    l3 = pm.TruncatedNormal('time-scale2', mu=20, sigma=5, lower=0)
    cov_f3 = pm.gp.cov.Periodic(input_dim=1, period=24, ls=l3)
    GP3 = pm.gp.Latent(mean_func=mean_f3, cov_func=cov_f3)
    #GP3 = pm.gp.Marginal(mean_f3, cov_f3)

    GP = GP1 + GP3

    f = GP.prior('f', X=timeIdx)
    #f = GP.marginal_likelihood("f", t_train.reshape(-1,1), y_train, noise = pm.HalfCauchy("noise",1))

    Lambda = np.mean(y_train) * Beta + tt.exp(f[:split]) + np.dot(decayKernel(y_train),y_train) * alpha
    #Lambda = np.mean(y_train) * Beta + f.exp() + np.dot(decayKernel(y_train), y_train) * alpha
    pm.Poisson('y_val', mu=Lambda, observed=y_train)
    trace = pm.sample(draws=10, tune=20, chains=1, target_accept=.9, random_seed=1, callback=my_callback)

par_dt = pd.DataFrame({
    'amplitude': trace['amplitude'],
    'time-scale': trace['time-scale'],
    'time-scale2': trace['time-scale2'],
    'amplification': trace['amplification'],
    'population_mean': trace['population_mean'],
})
par_dt.to_csv("par_dt.csv",index=False)

with model:
    val_samples = pm.sample_posterior_predictive(trace, random_seed=1)
forecasts_for_train = val_samples['y_val']  # 一个样本点一行

with model:
    #f = GP.conditional("f", t_test.reshape(-1,1))
    #Lambda = np.mean(y_test) * Beta + f.exp() + np.dot(decayKernel(y_test), y_test) * alpha
    Lambda = np.mean(y_test) * Beta + tt.exp(f[split:]) + np.dot(decayKernel(y_test),y_test) * alpha
    y_pred = pm.Poisson('y_pred', mu=Lambda, observed=y_test)
    test_samples = pm.sample_posterior_predictive(trace, var_names=['y_pred'], random_seed=1)

forecasts_for_test = test_samples['y_pred']
train_result_dt = pd.DataFrame(forecasts_for_train).T
train_result_dt.to_csv("train_result.csv",index=False)
test_result_dt = pd.DataFrame(forecasts_for_test).T
test_result_dt.to_csv("test_result.csv",index=False)

