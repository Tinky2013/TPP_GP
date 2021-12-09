import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano.tensor as tt
import time

def main():
    split = int(0.7 * len(y))

    y_train = y[:split]
    y_test = y[split:]

    timeIdx = np.arange(len(y_train) + len(y_test))[:, None]
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

        disMat = np.exp(- disMat)
        disMat[disMat>1] = 0

        return disMat

    with pm.Model() as model:
        # mean
        Mu = pm.HalfNormal('Mu', sigma=2)

        # Gaussian Process Prior

        mean_f3 = pm.gp.mean.Constant(c=0)
        a3 = pm.HalfNormal('amplitude3', sigma=2)
        gamma3 = pm.TruncatedNormal('time-scale3', mu=20, sigma=5, lower=0)
        cov_f3 = a3 ** 2 * pm.gp.cov.Periodic(input_dim=1, period=24, ls=gamma3)
        GP3 = pm.gp.Latent(mean_func=mean_f3, cov_func=cov_f3)

        # GP叠加
        GP = GP3
        f = GP.prior('f', X=timeIdx)

        # Decay Kernel
        alpha = pm.Gamma('occur', alpha=1, beta=1)
        beta = pm.Gamma('decay', alpha=1, beta=1)

        Lambda = Mu + tt.exp(f[:split]) + tt.dot(decayKernel(y_train)**beta, y_train) * alpha
        pm.Poisson('y_val', mu=Lambda, observed=y_train)
        trace = pm.sample(draws=500, tune=500, chains=1, target_accept=.9, random_seed=42, callback=my_callback)

    par_dt = pd.DataFrame({

        'amplitude3': trace['amplitude3'],
        'time-scale3': trace['time-scale3'],
        'occur': trace['occur'],
        'decay': trace['decay'],
        'Mu': trace['Mu'],
    })
    par_dt.to_csv("par_dt_"+save_path+".csv",index=False)

    with model:
        val_samples = pm.sample_posterior_predictive(trace, random_seed=42)
    forecasts_for_train = val_samples['y_val']  # 一个样本点一行

    with model:
        Lambda = Mu + tt.exp(f[split:]) + tt.dot(decayKernel(y_test)**beta, y_test) * alpha
        y_pred = pm.Poisson('y_pred', mu=Lambda, observed=y_test)
        test_samples = pm.sample_posterior_predictive(trace, var_names=['y_pred'], random_seed=42)

    forecasts_for_test = test_samples['y_pred']
    train_result_dt = pd.DataFrame(forecasts_for_train).T
    train_result_dt.to_csv("train_result_"+save_path+".csv",index=False)
    test_result_dt = pd.DataFrame(forecasts_for_test).T
    test_result_dt.to_csv("test_result_"+save_path+".csv",index=False)

df = pd.read_csv("data/data.csv")
y = df['sys3']
save_path = 'df3_2' # 每个跑实验改这个路径

if __name__ == '__main__':
    main()

