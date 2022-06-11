
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fbprophet


m = fbprophet.Prophet()


df = pd.read_csv("syn_data/trend8.csv")
df1 = pd.read_csv("syn_data/timestump.csv")
y = df['6']
t = df1['day']
split = int(0.667 * len(y))

y_train = y[:split]
y_test = y[split:]
t_train = t[:split]
t_test = t[split:]



df = pd.DataFrame([])

df['ds'] = t_train
df['y'] = y_train

m.fit(df)
future = m.make_future_dataframe(periods=len(t_test))
future.tail()
forecast = m.predict(future)
m.plot(forecast)

x1 = forecast['ds']
y1 = forecast['yhat']
plt.plot(x1,y1)
plt.show()

y_train_pred = y1[:split]
y_test_pred = y1[split:]

# train_result_dt = pd.DataFrame(y_train_pred)
# train_result_dt.to_csv("train_result_prophet.csv",index=False)
# test_result_dt = pd.DataFrame(y_test_pred)
# test_result_dt.to_csv("test_result_prophet.csv",index=False)

def MSE(y_true, y_pred):
    print(len(y_true), len(y_pred))
    return (np.square(y_true-y_pred).sum())/len(y_true)

def MAE(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / len(y_true)


print("training MSE:", MSE(y_train, y_train_pred),"testing MSE:", MSE(y_test, y_test_pred))
print("training MAE:", MAE(y_train, y_train_pred),"testing MAE:", MAE(y_test, y_test_pred))

