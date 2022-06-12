from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 【real world】 Double Periodic Kernel
df=pd.read_csv('data/data2.csv')
y=df['click']
t = np.arange(len(y))

plt.figure(figsize=(16,5))
plt.title('Data4')
plt.grid()
plt.xlabel('Time (in hours)')
plt.plot(t, y, 'black', lw=2, label='observed')
plt.savefig('dataDPK.jpg',dpi=600)

# 【synthetic】
# df=pd.read_csv('data/trend9.csv')
# plt.figure(figsize=(16,5))
# plt.title('Purchase Data with Long-term Trend (Non-linear Type 2)')
# plt.grid()
# plt.xlabel('Time (in hours)')
#
# for i in ['0','1','2','3','4','5','6','7','8']:
#     y=df[i]
#     t = np.arange(len(y))
#     plt.plot(t, y, 'silver', lw=1.5)
#
# y=df['9']
# #plt.plot(t, y, 'blue', lw=2)
# #plt.plot(t, y, 'darkcyan', lw=2)
# plt.plot(t, y, 'deeppink', lw=2)
#
# plt.savefig('syndt_trend9.jpg',dpi=600)