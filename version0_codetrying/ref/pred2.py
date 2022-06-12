import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano
import theano.tensor as tt
import time

df = pd.read_csv("data/click_count_hour.csv")
# 三列数据：VisitDateTime, click, pageload
y = df['click']