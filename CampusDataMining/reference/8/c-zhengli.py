# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 11:08:27 2019

@author: HP
"""

#from sklearn.cluster import KMeans
import pandas as pd
#import numpy as np
#from sklearn import metrics
#读取csv文件
#读取csv，取学号，成绩列
df = pd.read_csv('zy-cj2.csv')
p1 = pd.DataFrame(df)
print("------去除异常数据-----")
p1.drop_duplicates(subset=['xh','kch'],keep='first',inplace=True)
#p1.to_csv('zy-cj3.csv', index=False, encoding='utf-8')
print("------建立新表-----")
tj = p1['kch'].value_counts() #课程号为列号
#print(tj)
l1 = pd.DataFrame(tj)
l2 = l1.index.tolist() 

tj0 = p1['xh'].value_counts() #学号为行号
#print(tj)
l10 = pd.DataFrame(tj0)
l20 = l10.index.tolist() 

#建立新表df2，按照kch为列，xh为行
df2 = pd.DataFrame({'poetry_content': ''},index=[l20],columns=[l2])
#print(df2)
print("------存入数据-----")
for indexs in p1.index:
    p2 = p1.loc[indexs].values[0:-1]
    #df2['p2[1]','p2[0]']=p2[2]
    df2.loc[p2[0], p2[1]]=p2[2]
    #print(p2[1],p2[0])
df2.to_csv('zy-cj3.csv', index=False, encoding='utf-8')
    
