import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano.tensor as tt
import time

def main():
    split = int(0.667 * len(y))

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

    with pm.Model() as model:
        # Gaussian Process Prior
        # m = np.mean(y_train)
        Lambda0 = pm.Gamma('Lambda0', alpha=2, beta=2)

        # 多项式核
        # mu1 = pm.Normal('mu1', sigma=2)
        # mu1=0
        # mean_f1 = pm.gp.mean.Constant(c=mu1)
        # cov_f1 = pm.gp.cov.Polynomial(input_dim=1, c=0, d=2, offset=0)
        # GP1 = pm.gp.Latent(mean_func=mean_f1, cov_func=cov_f1)

        # SE核
        #mu2 = pm.Normal('mu2', sigma=10)
        mu2=0
        mean_f2 = pm.gp.mean.Constant(c=mu2)
        a2 = pm.HalfNormal('amplitude2', sigma=2)
        gamma2 = pm.TruncatedNormal('time-scale2', mu=10, sigma=5, lower=0)  # new hp
        cov_f2 = a2 ** 2 * pm.gp.cov.ExpQuad(input_dim=1, ls=gamma2)
        GP2 = pm.gp.Latent(mean_func=mean_f2, cov_func=cov_f2)

        # 周期核
        mean_f3 = pm.gp.mean.Constant(c=0)
        a3 = pm.HalfNormal('amplitude3', sigma=2)
        # l3 = pm.TruncatedNormal('time-scale3', mu=5, sigma=2, lower=0)
        l3 = pm.Gamma('time-scale3', alpha=2, beta=2)
        cov_f3 = a3 ** 2 * pm.gp.cov.Periodic(input_dim=1, period=7, ls=l3)
        GP3 = pm.gp.Latent(mean_func=mean_f3, cov_func=cov_f3)

        # 白噪声
        mean_fW = pm.gp.mean.Constant(c=0)
        cov_fW = pm.gp.cov.WhiteNoise(sigma=1)
        GPW = pm.gp.Latent(mean_func=mean_fW, cov_func=cov_fW)

        GP = GP2 + GP3 + GPW

        f = GP.prior('f', X=timeIdx)

        mu_tr = pm.Deterministic('mu_tr',Lambda0 * tt.exp(f[:split]))
        pm.Poisson('y_val', mu=mu_tr, observed=y_train)
        trace = pm.sample(draws=500, tune=500, chains=1, target_accept=.9, random_seed=42, callback=my_callback)

    par_dt = pd.DataFrame({
        'Lambda0': trace['Lambda0'],
        #'mu1': trace['mu1'],
        #'mu2': trace['mu2'],
        'amplitude2': trace['amplitude2'],
        'time-scale2': trace['time-scale2'],
        'amplitude3': trace['amplitude3'],
        'time-scale3': trace['time-scale3'],
    })
    par_dt.to_csv("par_dt_"+save_path+".csv",index=False)

    with model:
        val_samples = pm.sample_posterior_predictive(trace, random_seed=42)
    forecasts_for_train = val_samples['y_val']  # 一个样本点一行

    with model:
        mu_ts = pm.Deterministic('mu_ts', Lambda0 * tt.exp(f[split:]))
        y_pred = pm.Poisson('y_pred', mu=mu_ts, observed=y_test)
        test_samples = pm.sample_posterior_predictive(trace, var_names=['y_pred'], random_seed=42)

    forecasts_for_test = test_samples['y_pred']
    train_result_dt = pd.DataFrame(forecasts_for_train).T
    train_result_dt.to_csv("train_result_"+save_path+".csv",index=False)
    test_result_dt = pd.DataFrame(forecasts_for_test).T
    test_result_dt.to_csv("test_result_"+save_path+".csv",index=False)

if __name__ == '__main__':
    for i in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        df = pd.read_csv("syn_data/trend6.csv")
        y = df[i]
        save_path = 'trend6_'+i  # 每个跑实验改这个路径
        main()

