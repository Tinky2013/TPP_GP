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

    with pm.Model() as model:
        # Gaussian Process Prior
        m = np.mean(y_train)
        # Lambda0 = pm.Gamma('Lambda0', alpha=2, beta=2)
        Lambda0 = pm.HalfNormal('Lambda0', sigma=2)

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
        GP2 = pm.gp.Marginal(mean_func=mean_f2, cov_func=cov_f2)

        # 周期核
        mean_f3 = pm.gp.mean.Constant(c=0)
        l3 = pm.TruncatedNormal('time-scale3', mu=20, sigma=5, lower=0)
        cov_f3 = pm.gp.cov.Periodic(input_dim=1, period=24, ls=l3)
        GP3 = pm.gp.Marginal(mean_func=mean_f3, cov_func=cov_f3)

        # 白噪声
        sigma = pm.HalfNormal(name='sigma', sigma=10)



        GP = GP2 + GP3

        # Likelihood.
        y_pred = GP.marginal_likelihood('y_pred', X=t_train.reshape(-1,1), y=y_train, noise=sigma)
        # Sample.
        trace = pm.sample(draws=200, chains=1, tune=200)


    par_dt = pd.DataFrame({
        'Lambda0': trace['Lambda0'],
        #'mu1': trace['mu1'],
        #'mu2': trace['mu2'],
        'amplitude2': trace['amplitude2'],
        'time-scale2': trace['time-scale2'],
        'time-scale3': trace['time-scale3'],
    })
    par_dt.to_csv("par_dt_"+save_path+".csv",index=False)

    with model:
        x_train_conditional = GP.conditional('x_train_conditional', t_train.reshape(-1,1))
        y_train_pred_samples = pm.sample_posterior_predictive(trace, var_names=['x_train_conditional'], samples=100)

        x_test_conditional = GP.conditional('x_test_conditional', t_test.reshape(-1,1))
        y_test_pred_samples = pm.sample_posterior_predictive(trace, var_names=['x_test_conditional'], samples=100)

    forecasts_for_train = y_train_pred_samples['y_pred']
    forecasts_for_test = y_test_pred_samples['y_pred']
    train_result_dt = pd.DataFrame(forecasts_for_train).T
    train_result_dt.to_csv("train_result_"+save_path+".csv",index=False)
    test_result_dt = pd.DataFrame(forecasts_for_test).T
    test_result_dt.to_csv("test_result_"+save_path+".csv",index=False)

df = pd.read_csv("data/click_count_hour.csv")
y = df['click'][:100]
save_path = 'df7_1' # 每个跑实验改这个路径

if __name__ == '__main__':
    main()

