import numpy as np
import pandas as pd

# single periodic
# sample = []
# for n in range(10):
#     trajectory = []
#     for i in range(180):
#         if i%7 == 5 or i%7 == 6:
#             trajectory.append(int(np.mean(np.random.poisson(10,size=5))))
#         elif i%7 == 5:
#             trajectory.append(int(np.mean(np.random.poisson(8,size=5))))
#         else:
#             trajectory.append(int(np.mean(np.random.poisson(5,size=5))))
#     sample.append(trajectory)
#
# sample = pd.DataFrame(sample).transpose()
# sample.to_csv('sinper3.csv',index=False)

# double periodic
# sample = []
# for n in range(10):
#     trajectory = []
#     # 月份的周期
#     mu = [15, 15, 15, 14, 14, 14, 13, 13, 13,
#           12, 12, 12, 11, 11, 11, 10, 10, 10,
#           9, 9, 9, 8, 8, 8, 7, 7, 7, 6, 6, 6]
#     for i in range(180):
#         mu0 = mu[i%30]
#         if i%7 == 5 or i%7 == 6:
#             trajectory.append(int(np.mean(np.random.poisson(mu0,size=5))))
#         # elif i%7 == 5:
#         #     trajectory.append(int(np.mean(np.random.poisson(mu0,size=5))*0.8))
#         else:
#             trajectory.append(int(np.mean(np.random.poisson(mu0,size=5))/2))
#     sample.append(trajectory)
#
# sample = pd.DataFrame(sample).transpose()
# sample.to_csv('double1.csv',index=False)/home/yang/PycharmProjects/DLRL/combine/

# specific trend
sample = []
for n in range(10):
    trajectory = []
    # 潜在趋势（trend1-2）
    # mu = [10]*15+[11]*15+[12]*15+[13]*15+[14]*15+[15]*30+[14]*15+[13]*15+[12]*15+[11]*15+[10]*15
    # mu = [5]*15+[7]*15+[9]*15+[11]*15+[13]*15+[15]*30+[13]*15+[11]*15+[9]*15+[7]*15+[5]*15

    #（3-6）
    #mu = [10]*10+[11]*10+[12]*10+[13]*10+[14]*10+[15]*30+[14]*20+[13]*20+[12]*20+[11]*20+[10]*20
    #mu = [10]*5+[11]*5+[12]*5+[13]*5+[14]*5+[15]*30+[14]*25+[13]*25+[12]*25+[11]*25+[10]*25
    #mu = [5]*10+[7]*10+[9]*10+[11]*10+[13]*10+[15]*30+[13]*20+[11]*20+[9]*20+[7]*20+[5]*20
    #mu = [5]*5+[7]*5+[9]*5+[11]*5+[13]*5+[15]*30+[13]*25+[11]*25+[9]*25+[7]*25+[5]*25

    #（8-10）
    mu = [5]*15+[10]*15+[14]*15+[17]*15+[19]*15+[20]*30+[19]*15+[17]*15+[14]*15+[10]*15+[10]*15
    #mu = [5]*10+[10]*10+[14]*10+[17]*10+[19]*10+[20]*30+[19]*20+[17]*20+[14]*20+[10]*20+[10]*20

    #mu = [5]*5+[10]*5+[14]*5+[17]*5+[19]*5+[20]*30+[19]*25+[17]*25+[14]*25+[10]*25+[10]*25

    for i in range(180):
        if i%7 == 5 or i%7 == 6:
            trajectory.append(int(np.mean(np.random.poisson(mu[i],size=5))))
        else:
            trajectory.append(int(np.mean(np.random.poisson(mu[i],size=5))*0.75))
    sample.append(trajectory)

sample = pd.DataFrame(sample).transpose()
sample.to_csv('trend8.csv',index=False)


