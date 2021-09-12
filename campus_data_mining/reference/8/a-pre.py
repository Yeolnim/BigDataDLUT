# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 16:59:36 2019

@author: HP
"""
 
#from sklearn.cluster import KMeans
import pandas as pd
#import numpy as np
#from sklearn import metrics
#读取csv文件
#读取csv，取学号，成绩列
df = pd.read_csv('bks_cjxx_out.csv',usecols=[0,4,10])
p1 = pd.DataFrame(df)
print("------数据清洗-----")
df.drop_duplicates()#删除重复值
df.dropna()#空值舍去
print("------插入专业-----")
df.insert(3,'zym',df['xh'].astype(int))#插入课程号列
#df['zym'] = df['zym'].astype(int)
#print(df)
print("------取学生最多专业-----")
df2 = pd.read_csv('bks_xjjbsjxx_out.csv',usecols=[0,2])
#print(df2)
df2 = df2.drop_duplicates(subset=['xh'], keep='first')
z = df2.groupby(['zym']).size()
z = pd.DataFrame(z)
z2 = z[0].max()
z3 = z[z.iloc[:,0] == z2].index.tolist() 
#print (z3)
z4 = z3[0]
print("------生成词典-----")
df12 = df2.loc[df2['zym'] == z4]
p2 = pd.DataFrame(df12)
d2 = p2.set_index('xh').T.to_dict('list')
#print(d2)
print("------开始替换-----")
df['zym'].replace(d2,inplace=True) #将zym替换成专业True
print("------开始筛选-----")
df20 = df.loc[df['zym'] == z4]
df20.to_csv('zy-cj.csv', index=False, encoding='utf-8')
