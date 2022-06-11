import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano.tensor as tt
import time

class DataLoader():
    def __init__(self):
        df = pd.read_csv("data/click_count_hour.csv")
        self.y = df['click'][:100]
        self.split = int(0.7 * len(self.y))

        self.y_train = self.y[:self.split]
        self.y_test = self.y[self.split:]

        self.timeIdx = np.arange(len(self.y_train) + len(self.y_test))[:, None]
        t = np.arange(len(self.y_train) + len(self.y_test))

        self.t_train = t[:self.split]
        self.t_test = t[self.split:]
