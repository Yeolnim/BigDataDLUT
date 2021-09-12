# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 15:05:18 2019

@author: HP
"""
#导入依赖库
import pandas as pd
import numpy as np
data = pd.read_excel("last.xlsx", sheet_name="Sheet1")
#print(data)
print("------将课程号转化成课程名称-----")
#将课程号转化成课程名称
df2 = pd.read_csv('bks_kcsjxx_out.csv',usecols=[0,1])
#print(df2)
d2 = df2.set_index('kch').T.to_dict('list')
#print(d2)
#data.iloc[0,].replace(d2, inplace=True)#将name替换成课程名称
#print(data.iloc[0,])
data2 = data.iloc[:,0].astype(str)
data2.replace(d2, inplace=True)#将name替换成课程名称
#print(data2)
data.insert(0,'data2',data2)

data3 = data.columns.values.tolist()
data3 = pd.DataFrame(data3)
data3.replace(d2, inplace=True)
dt = np.array(data3)
#print(dt)
dt2 = dt.flatten()
dt3 = dt2.tolist()
#print(dt3)
data.loc['new'] = dt3
data.to_excel('out.xlsx',index=True)
