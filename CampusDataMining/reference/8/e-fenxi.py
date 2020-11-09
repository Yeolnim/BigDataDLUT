# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 15:05:18 2019

@author: HP
"""
#导入依赖库
import pandas as pd

data = pd.read_csv('zy-cj4.csv')
dataframe1 = pd.DataFrame(data)
data2 = dataframe1.corr() #求pearson相关系数
#data3 = data2.replace(1.000000,0) #将1值替换
#data3 = pd.DataFrame(data3)
data3 = pd.DataFrame(data2)
data3.to_csv('last.csv', index=False, encoding='utf-8')
data3.to_excel('last.xlsx',index=True)
