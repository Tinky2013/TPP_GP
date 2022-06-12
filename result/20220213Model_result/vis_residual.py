import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

df=pd.read_csv('train_estimate.csv')
y=df['data3']
yhat=df['full3']
res=y-yhat
t = np.arange(len(y))

plt.figure(figsize=(16,5))
plt.ylim((-3,3))
plt.hlines(y=0, xmin=0, xmax=357, colors="black", linestyles="dashed")
plt.scatter(t,res,c='orange', alpha=0.5, edgecolors='black')

plt.savefig('res_full_d3.jpg',dpi=600)
#plt.show()

