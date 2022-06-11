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
        # m = np.mean(y_train)
        Lambda0 = pm.Gamma('Lambda0', alpha=2, beta=2)

        pm.Poisson('y_val', mu=Lambda0, observed=y_train)
        trace = pm.sample(draws=500, tune=500, chains=1, target_accept=.9, random_seed=42, callback=my_callback)

    par_dt = pd.DataFrame({
        'Lambda0': trace['Lambda0'],
        #'mu1': trace['mu1'],
        #'mu2': trace['mu2'],
        #'amplitude2': trace['amplitude2'],
        #'time-scale2': trace['time-scale2'],
        #'time-scale3': trace['time-scale3'],
    })
    par_dt.to_csv("par_dt_"+save_path+".csv",index=False)

    with model:
        val_samples = pm.sample_posterior_predictive(trace, random_seed=42)
    forecasts_for_train = val_samples['y_val']  # 一个样本点一行

    with model:

        y_pred = pm.Poisson('y_pred', mu=Lambda0, observed=y_test)
        test_samples = pm.sample_posterior_predictive(trace, var_names=['y_pred'], random_seed=42)

    forecasts_for_test = test_samples['y_pred']
    train_result_dt = pd.DataFrame(forecasts_for_train).T
    train_result_dt.to_csv("train_result_"+save_path+".csv",index=False)
    test_result_dt = pd.DataFrame(forecasts_for_test).T
    test_result_dt.to_csv("test_result_"+save_path+".csv",index=False)

df = pd.read_csv("data/data3.csv")
y = df['click']
save_path = 'df3_hpoi' # 每个跑实验改这个路径

if __name__ == '__main__':
    main()

