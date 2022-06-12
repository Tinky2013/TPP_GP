import numpy as np
import pandas as pd
from PyEMD import EEMD, EMD, CEEMDAN
import pymc3 as pm
import matplotlib.pyplot as plt

df1 = pd.read_csv('data/data7.csv')
y1 = np.array(df1['click'])
t = np.arange(len(y1))

ceemdan = CEEMDAN()

eIMFs = ceemdan.ceemdan(y1)
nIMFs = eIMFs.shape[0]

plt.figure(figsize=(12,9))
plt.subplot(nIMFs+1, 1, 1)
plt.plot(t, y1, 'r')

# p = eemd.get_imfs_and_residue()
# res = p[-1]

rec = np.array([0]*len(y1))
for n in range(nIMFs):
    for i in range(len(y1)):
        rec[i] += eIMFs[n][i]
# for i in range(len(y1)):
#     rec[i]+=res[i]

for n in range(nIMFs):
    plt.subplot(nIMFs+3, 1, n+2)
    plt.plot(t, eIMFs[n], 'g')
    plt.ylabel("eIMF %i" %(n+1))
    plt.locator_params(axis='y', nbins=5)

# plt.subplot(nIMFs+4, 1, nIMFs+2)
# plt.plot(t, res, 'g')
# plt.ylabel("res")
plt.subplot(nIMFs+3, 1, nIMFs+2)
plt.plot(t, y1, 'b')
plt.ylabel("ori")
plt.subplot(nIMFs+3, 1, nIMFs+3)
plt.plot(t, rec, 'r')
plt.ylabel("rec")

plt.xlabel("Time [s]")
plt.tight_layout()
plt.savefig('./emd.png', dpi=120)
plt.close()