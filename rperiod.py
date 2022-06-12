import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from robustperiod import robust_period, robust_period_full, plot_robust_period

y = np.array(pd.read_csv('data/data7.csv')['click'])

plt.plot(y)
plt.title('Dummy dataset')
plt.show()

lmb = 1e+6
c = 2
num_wavelets = 8
zeta = 1.345

periods, W, bivar, periodograms, p_vals, ACF = robust_period_full(
    y, 'db10', num_wavelets, lmb, c, zeta)
plot_robust_period(periods, W, bivar, periodograms, p_vals, ACF)