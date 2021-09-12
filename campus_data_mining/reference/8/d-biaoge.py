# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 12:23:18 2019

@author: HP
"""
#from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
#from sklearn import metrics
#读取csv文件
#读取csv，取学号，成绩列
df = pd.read_csv('zy-cj3.csv')
p1 = pd.DataFrame(df)
print("------规范格式-----")
df['Col_sum'] = df.apply(lambda x: x.sum(), axis=1)
df.sort_values(by="Col_sum" , ascending=False)

df2=df.copy()
#print(df2)
a=0
b=10
for a in range(b):
    #df2[0:1]=df2[0:1].fillna('null')
    #print(df2)
    cols=[x for i,x in enumerate(df2.columns) if np.isnan(df2.iat[a,i])]
    #print(cols)
    df2=df2.drop(cols,axis=1)
    a=a+1
#print(df2)
#df3 = df2.dropna(axis=0) #删除表中含有任何NaN的行
#print(df3)
df3 = df2.fillna(method='ffill', axis=1)#对NAN采用向前填充
#print(df3)
df3= df3.drop('Col_sum', 1)
df3.to_csv('zy-cj4.csv', index=False, encoding='utf-8')
