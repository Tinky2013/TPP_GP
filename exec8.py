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
y_train, y_test = y[:split], y[split:]
timeIdx = dataLoader.timeIdx
t = np.arange(len(y_train) + len(y_test))
t_train, t_test = t[:split], t[split:]


def main():
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

        disMat = np.exp(- disMat)
        disMat[disMat>1] = 0

        return disMat

    with pm.Model() as model:
        # mean
        Mu = pm.HalfNormal('Mu', sigma=2)

        # Gaussian Process Prior
        m = pm.Normal('mean', sigma=2)

        mean_f1 = pm.gp.mean.Constant(c=m)
        a1 = pm.HalfNormal('amplitude1', sigma=2)
        gamma1 = pm.TruncatedNormal('time-scale1', mu=2.5, sigma=5, lower=0)    # new hp
        cov_f1 = a1 ** 2 * pm.gp.cov.ExpQuad(input_dim=1, ls=gamma1)
        GP1 = pm.gp.Latent(mean_func=mean_f1, cov_func=cov_f1)

        mean_f3 = pm.gp.mean.Constant(c=0)
        a3 = pm.HalfNormal('amplitude3', sigma=2)
        gamma3 = pm.TruncatedNormal('time-scale3', mu=20, sigma=5, lower=0)
        cov_f3 = a3 ** 2 * pm.gp.cov.Periodic(input_dim=1, period=24, ls=gamma3)
        GP3 = pm.gp.Latent(mean_func=mean_f3, cov_func=cov_f3)

        mean_fW = pm.gp.mean.Constant(c=0)
        cov_fW = pm.gp.cov.WhiteNoise(sigma=1)
        GPW = pm.gp.Latent(mean_func=mean_fW, cov_func=cov_fW)

        # GP叠加
        GP = GP1 + GP3 + GPW
        f = GP.prior('f', X=timeIdx)

        # Decay Kernel
        alpha = pm.Gamma('occur', alpha=1, beta=1)
        beta = pm.Gamma('decay', alpha=1, beta=1)

        Lambda = Mu + tt.exp(f[:split]) + tt.dot(decayKernel(y_train)**beta, y_train) * alpha
        pm.Poisson('y_val', mu=Lambda, observed=y_train)
        trace = pm.sample(draws=500, tune=500, chains=1, target_accept=.9, random_seed=1, callback=my_callback)

    par_dt = pd.DataFrame({
        'mean': trace['mean'],
        'Mu': trace['Mu'],
        'amplitude1': trace['amplitude1'],
        'amplitude3': trace['amplitude3'],
        'time-scale1': trace['time-scale1'],
        'time-scale3': trace['time-scale3'],
        'occur': trace['occur'],
        'decay': trace['decay'],
    })
    par_dt.to_csv("par_dt_"+save_path+".csv",index=False)

    with model:
        val_samples = pm.sample_posterior_predictive(trace, random_seed=1)
    forecasts_for_train = val_samples['y_val']  # 一个样本点一行

    with model:
        Lambda = Mu + tt.exp(f[split:]) + tt.dot(decayKernel(y_test)**beta, y_test) * alpha
        y_pred = pm.Poisson('y_pred', mu=Lambda, observed=y_test)
        test_samples = pm.sample_posterior_predictive(trace, var_names=['y_pred'], random_seed=1)

    forecasts_for_test = test_samples['y_pred']
    train_result_dt = pd.DataFrame(forecasts_for_train).T
    train_result_dt.to_csv("train_result_"+save_path+".csv",index=False)
    test_result_dt = pd.DataFrame(forecasts_for_test).T
    test_result_dt.to_csv("test_result_"+save_path+".csv",index=False)

save_path = '8' # 每个跑实验改这个路径

if __name__ == '__main__':
    main()

