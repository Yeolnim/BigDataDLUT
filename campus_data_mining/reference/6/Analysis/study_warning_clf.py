#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd

total_data = pd.read_csv('../Data/total1.csv', encoding='utf-8')
print(total_data.head())
print(total_data.columns)


# In[41]:


cnt1 = 0
cnt2 = 0
cnt3 = 0
for index in total_data.index:
    xh = str(total_data.at[index, 'xh'])
    if abs(total_data.at[index, 'xueyejinggao1'] - 2) < 1e-3 and xh[3] == '5':
        cnt1 += 1
    if abs(total_data.at[index, 'xueyejinggao1'] - 2) < 1e-3 and xh[3] == '6':
        cnt2 += 1
    if abs(total_data.at[index, 'xueyejinggao2'] - 2) < 1e-3:
        cnt3 += 1
print(cnt1)
print(cnt2)
print(cnt3)


# In[48]:


import numpy as np


train_x = []
train_y = []
test_x = []
test_y = []

feature  = ['xb', 'mean', 'var', 'chifan', 'time', 'morning', 'afternoon',
           'duanshipin_time', 'shipin_time', 'zhibo_time',
           'youxi_time', 'tengxun_time', 'duanshipin_period', 'shipin_period',
           'zhibo_period', 'youxi_period', 'tengxun_period']

for index in total_data.index:
    xh = str(total_data.at[index, 'xh'])
    feature_array = np.random.rand(len(feature))

    if xh[3] == '5' or xh[3] == '6':
        feature_map = feature
        for col in feature_map:
            feature_array[feature_map.index(col)] = total_data.at[index, col]
        train_x.append(feature_array)
        train_y.append(total_data.at[index, 'xueyejinggao1'])
    elif xh[3] == '7':
        feature_map = feature
        for col in feature_map:
            feature_array[feature_map.index(col)] = total_data.at[index, col]
        test_x.append(feature_array)
        test_y.append(total_data.at[index, 'xueyejinggao2'])

cnt1 = 0
cnt2 = 0
for item in train_y:
    if abs(item - 2) < 1e-3:
        cnt1 += 1
for item in test_y:
    if abs(item - 2) < 1e-3:
        cnt2 += 1
print(cnt1, len(train_y))
print(cnt2, len(test_y))  


# In[49]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
train_x = ss.fit_transform(train_x)
test_x = ss.transform(test_x)

train_y = np.array(train_y).reshape(-1, 1)
test_y = np.array(test_y).reshape(-1, 1)


# In[100]:


from sklearn.svm import SVC

clf = SVC(C=3.5, gamma=0.006)
clf.fit(train_x, train_y.ravel())

print(clf.score(train_x, train_y.ravel()))
print(clf.score(test_x, test_y.ravel()))

from sklearn.metrics import confusion_matrix

train_predict_y_svm = clf.predict(train_x)
cm_1 = confusion_matrix(train_y.ravel(), train_predict_y_svm)

test_predict_y_svm = clf.predict(test_x)
cm_2 = confusion_matrix(test_y.ravel(), test_predict_y_svm)
print(cm_1)
print(cm_1[1][1] * 1.0 / (cm_1[1][1] + cm_1[1][0]), cm_1[1][1] * 1.0 / (cm_1[1][1] + cm_1[0][1]))
print(cm_2)
print(cm_2[1][1] * 1.0 / (cm_2[1][1] + cm_2[1][0]), cm_2[1][1] * 1.0 / (cm_2[1][1] + cm_2[0][1]))


# In[51]:


cnt = 0
for item in train_y:
    if abs(item[0] - 2) < 1e-3:
        cnt += 1
print(cnt)
print(train_y.shape)

cnt = 0
for item in test_y:
    if abs(item[0] - 2) < 1e-3:
        cnt += 1
print(cnt)
print(test_y.shape)


# In[101]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=20, min_samples_split=20)
clf.fit(train_x, train_y.ravel())

print(clf.score(train_x, train_y.ravel()))
print(clf.score(test_x, test_y.ravel()))

train_predict_y_rf = clf.predict(train_x)
cm_1 = confusion_matrix(train_y.ravel(), train_predict_y_rf)

test_predict_y_rf = clf.predict(test_x)
cm_2 = confusion_matrix(test_y.ravel(), test_predict_y_rf)
print(cm_1)
print(cm_1[1][1] * 1.0 / (cm_1[1][1] + cm_1[1][0]), cm_1[1][1] * 1.0 / (cm_1[1][1] + cm_1[0][1]))
print(cm_2)
print(cm_2[1][1] * 1.0 / (cm_2[1][1] + cm_2[1][0]), cm_2[1][1] * 1.0 / (cm_2[1][1] + cm_2[0][1]))


# In[103]:


train_predict_y = []
test_predict_y = []

for i in range(len(train_predict_y_rf)):
    train_predict_y.append(min(train_predict_y_rf[i], train_predict_y_svm[i]))
for j in range(len(test_predict_y_svm)):
    test_predict_y.append(min(test_predict_y_rf[j], test_predict_y_svm[j]))

cm_1 = confusion_matrix(train_y.ravel(), train_predict_y)

cm_2 = confusion_matrix(test_y.ravel(), test_predict_y)
print(cm_1)
print(cm_1[1][1] * 1.0 / (cm_1[1][1] + cm_1[1][0]), cm_1[1][1] * 1.0 / (cm_1[1][1] + cm_1[0][1]))
print(cm_2)
print(cm_2[1][1] * 1.0 / (cm_2[1][1] + cm_2[1][0]), cm_2[1][1] * 1.0 / (cm_2[1][1] + cm_2[0][1]))


# In[ ]:




