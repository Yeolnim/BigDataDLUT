# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 18:51:59 2019

@author: guanlan
"""

#第一问将数据读入，然后将其按不同学生学号提取出来后，再设置为ndarray类型
#将每个Student都设置为一个类实体
class Student(object):
    def __init__(self,grade):
        self.grade = grade
    def chengji(self,*chengji):
        self.chengji = chengji
    def kecheng(self,*kecheng):
        self.kecheng = kecheng
    def kecheng_data(self,date):
        self.kecheng_data = data
    def rixue_xueqi(self,xueqi):
        self.ruxue_xueqi = xueqi
    def xueqi_chengji(self,xueqi):
        xueqi_chengji=self.chengji[xueqi-1]
        xueqi_kecheng=self.kecheng[xueqi-1]
        return xueqi_chengji
#读入文件
import pandas as pd
cj = pd.read_csv('bks_cjxx1.csv')
#cj为dataframe结构，将学号提取出来之后转换为list类型
yuanshi_xuehao = cj['xh'].tolist()
#set去重
xuehao = set(yuanshi_xuehao)        
xuehao = list(xuehao)
l=len(yuanshi_xuehao)
import numpy as np
gama = 0.9
train_number = int(gama*l)
index_1 = np.random.choice(l,train_number,replace=False)
#print(index_1)
chengji = cj['kccj']
kecheng = cj['kch']
xn = cj['xn']
xqm = cj['xqm']
#print(type(ndarray_chengji))
list_chengji=list(chengji)
list_kecheng=list(kecheng)
X=[]
y=[]
X1=[]
X2=[]
count = 1
for i in range(len(xuehao)):
    student_grade = xuehao[i-1]
    ruxue_nianfen = int(student_grade/100000)
    #创建实例
    student_grade = Student(xuehao[i-1])
    #把这个学号的所有成绩索引出来：index2为一个列表，但对于一个列表不可以直接索引另外一个列表
    index_2 = [x for x in range(l) if yuanshi_xuehao[x] == student_grade.grade]
    #创建列表，把成绩分门别类放进去
    grade_chengji=[[],[],[],[],[],[],[],[],[],[],[],[]]
    grade_kecheng=[[],[],[],[],[],[],[],[],[],[],[],[]]
    xueqi=[]
    for i in index_2:
        #grade_kecheng.append(list_kecheng[i])
        kecheng_xn = xn[i]
        kecheng_xn = int(kecheng_xn[:4])
        #print(kecheng_xn)
        kecheng_xqm = int(xqm[i])
        #print(kecheng_xqm)
        kecheng_xueqi =(kecheng_xn-ruxue_nianfen)*2+kecheng_xqm
        xueqi.append(kecheng_xueqi)
        #print(kecheng_xueqi)
        grade_kecheng[kecheng_xueqi-1].append(list_kecheng[i])
        grade_chengji[kecheng_xueqi-1].append(list_chengji[i])
    a=max(xueqi)
    student_grade.xueqi = a
    student_grade.kecheng=grade_kecheng[:a]
    student_grade.chengji=grade_chengji[:a]
    count = count+1
    if a==4:
        one_chengji=student_grade.xueqi_chengji(0)
        two_chengji=student_grade.xueqi_chengji(1)
        three_chengji=student_grade.xueqi_chengji(2)
        four_chengji=student_grade.xueqi_chengji(3)
        if len(three_chengji)==0|len(four_chengji)==0:
            break
        #average_one_chengji = sum(one_chengji)/len(one_chengji)
        #average_two_chengji = sum(two_chengji)/len(two_chengji)
        average_three_chengji = sum(three_chengji)/len(three_chengji)
        max_three_chengji = max(three_chengji)
        average_four_chengji = sum(four_chengji)/len(four_chengji) 
        #min.max
        X1.append(average_three_chengji)
        X2.append(max_three_chengji)
        #X2.append(average_one_chengji)
        y.append(average_four_chengji)
        
X=[X1,X2]
print(l)
print(len(xuehao))
print(count)
X=np.array(X)
y=np.array(y)
X=X.T
y=y.T
print("_________________")
print(y)
print(X)
max([1,2,3])

#通用函数
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
        plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
        plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")

#线性回归部分
import numpy as np
#闭式解算出来的线性回归
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

lin_reg.fit(X, y)
print(lin_reg.intercept_, lin_reg.coef_)

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X[0]#表示之前的特征集
X_poly[0]#表示二次特征集，下面将用二次特征集作为线性函数的输入去拟合
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_

plot_learning_curves(lin_reg, X, y)


#随机梯度下降：在每一步随机选择一个实例
l=len(X1)
#生成偏置项
X_b = np.c_[np.ones((len(X), 1)), X]

n_epochs = 100
t0, t1 = 5, 50 # learning schedule hyperparameters

def learning_schedule(t):
    return t0 / (t + t1)
theta = np.random.randn(3,1) # random initialization
for epoch in range(n_epochs):
    for i in range(l):
        random_index = np.random.randint(l)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * l + i)
        theta = theta - eta * gradients
        
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(n_iter=50, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())

#决策树
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)
tree_reg.predict([[55, 65]])











