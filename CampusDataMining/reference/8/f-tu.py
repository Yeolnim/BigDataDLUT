# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:37:49 2019

@author: HP
"""

import pandas as pd
train = pd.read_csv("last.csv")

import seaborn as sns
import matplotlib.pyplot as plt
def showcov(df):
    #dfData = df.corr()
    dfData = df
    plt.subplots(figsize=(18, 18)) # 设置画面大小
    sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="YlGnBu")
    plt.savefig('./BluesStateRelation.png')
    plt.show()

showcov(train)