# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 10:27:11 2019

@author: HP
"""

#from sklearn.cluster import KMeans
import pandas as pd
#import numpy as np
#from sklearn import metrics
#读取csv文件
#读取csv，取学号，成绩列
df = pd.read_csv('zy-cj.csv',usecols=[0,1,2])
p1 = pd.DataFrame(df)
print("------按学员处理-----")
tj = df['xh'].value_counts()
#print(tj)
tj2=sorted(tj.items(), key=lambda x: x[1],reverse=True)
d1=dict(tj2)#将返回的列表强转为字典
#print(d1)
df.insert(3,'cs',df['xh'])#插入课程号列
#df.insert(4,'name',df['kch'])#插入课程号列
df['cs'].replace(d1, inplace=True)#将cs替换成出现次数
#print(df)
df=df.loc[df['cs'] > 20]#统计学员参与20门课程以上
print("------按课程处理-----")
tj0 = df['kch'].value_counts()
#print(tj0)
tj20=sorted(tj0.items(), key=lambda x: x[1],reverse=True)
d10=dict(tj20)#将返回的列表强转为字典
#print(d10)
df.insert(4,'cs0',df['kch'])#插入课程号列
#df.insert(4,'name',df['kch'])#插入课程号列
df['cs0'].replace(d10, inplace=True)#将cs替换成出现次数
#print(df)
df=df.loc[df['cs0'] > 100]#统计学员参与20门课程以上
#dfz = df.sort_index(by=["xh","kch","kccj"],ascending = [False,False,False]) 
dfz = df.sort_values(by=["xh","kch","kccj"],axis='index',ascending = [False,False,False]) 
dfz.to_csv('zy-cj2.csv', index=False, encoding='utf-8')