#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

total_data = pd.read_csv('../Data/total1.csv', encoding='utf-8')
print(total_data.head())
print(total_data.columns)


# In[2]:


score_col = [ 'bx', 'xx', 'dbx', 'xbx']
cosume_col = ['chifan','time', 'mean', 'var']
course_col = ['morning', 'afternoon']
app_col = [ 'duanshipin_time', 'shipin_time', 'zhibo_time',
           'youxi_time', 'tengxun_time', 'duanshipin_period', 'shipin_period',
           'zhibo_period', 'youxi_period', 'tengxun_period']
gender_col = ['xb']


# In[3]:


import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('pylab', 'inline')


corr_matrix = np.random.rand(len(score_col), len(cosume_col))
for col1 in cosume_col:
    for col2 in score_col:
        print('corrcoef ', col1, ' ', col2, ':', np.corrcoef(total_data[col1], total_data[col2])[0, 1])
        corr_matrix[score_col.index(col2), cosume_col.index(col1)] = np.corrcoef(total_data[col1], total_data[col2])[0, 1]
print(corr_matrix)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr_matrix, interpolation='nearest', cmap='Oranges')
fig.colorbar(cax)
ax.set_xticklabels([''] + cosume_col)
ax.set_yticklabels([''] + score_col)
plt.show()


rf_matrix = np.random.rand(len(score_col), len(cosume_col))
for col in score_col:
    rf = RandomForestRegressor(n_estimators=20)
    rf.fit(total_data[cosume_col], total_data[col])
    print(rf.feature_importances_)
    rf_matrix[score_col.index(col)] = rf.feature_importances_
print(rf_matrix)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(rf_matrix, interpolation='nearest', cmap='Oranges')
fig.colorbar(cax)
ax.set_xticklabels([''] + cosume_col)
ax.set_yticklabels([''] + score_col)
plt.show()


# In[4]:


for col1 in cosume_col:
    for col2 in score_col:
        cosume_mean = []
        cosume_mean.append(total_data.loc[total_data[col2] < 60, col1])
        cosume_mean.append(total_data.loc[(total_data[col2] >= 60) & (total_data[col2] < 70), col1])
        cosume_mean.append(total_data.loc[(total_data[col2] >= 70) & (total_data[col2] < 80), col1])
        cosume_mean.append(total_data.loc[(total_data[col2] >= 80) & (total_data[col2] < 90), col1])
        cosume_mean.append(total_data.loc[total_data[col2] >= 90, col1])
        plt.boxplot(
            x=cosume_mean, 
            labels=['0-60','60-70','70-80','80-90','90-100'],
            medianprops={'color':'orange'},
            boxprops=dict(color="blue"),
            whiskerprops = {'color': "black"},
            capprops = {'color': "red"},
            flierprops={'color':'purple','markeredgecolor':"purple"}
        )
        plt.xlabel(col2 + ' score')
        plt.ylabel('cosume ' + col1)
        plt.title(col2 + ' - cosume ' + col1)
        plt.show()


# In[5]:


for col in cosume_col:
    list_male = total_data.loc[total_data['xb'] == 1, col]
    list_female = total_data.loc[total_data['xb'] == 2, col]
    
    plt.hist(list_male, rwidth=0.9, bins=20, range=(0, max(list_male)), label='male')
    plt.hist(list_female, rwidth=0.9, bins=20, range=(0, max(list_male)), label='female')
    plt.xlabel(col)
    plt.ylabel('numbers')
    plt.title('gender-'+col)
    plt.legend()
    plt.show()


# In[6]:


corr_matrix = np.random.rand(len(score_col), len(course_col))
for col1 in course_col:
    for col2 in score_col:
        print('corrcoef ', col1, ' ', col2, ':', np.corrcoef(total_data[col1], total_data[col2])[0, 1])
        corr_matrix[score_col.index(col2), course_col.index(col1)] = np.corrcoef(total_data[col1], total_data[col2])[0, 1]
print(corr_matrix)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr_matrix, interpolation='nearest', cmap='Oranges')
fig.colorbar(cax)
ax.set_xticklabels([''] + course_col)
ax.set_yticklabels([''] + score_col)
plt.show()


rf_matrix = np.random.rand(len(score_col), len(course_col))
for col in score_col:
    rf = RandomForestRegressor(n_estimators=20)
    rf.fit(total_data[course_col], total_data[col])
    print(rf.feature_importances_)
    rf_matrix[score_col.index(col)] = rf.feature_importances_
print(rf_matrix)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(rf_matrix, interpolation='nearest', cmap='Oranges')
fig.colorbar(cax)
ax.set_xticklabels([''] + course_col)
ax.set_yticklabels([''] + score_col)
plt.show()


# In[7]:


for col1 in course_col:
    for col2 in score_col:
        course_mean = []
        course_mean.append(total_data.loc[total_data[col2] < 60, col1])
        course_mean.append(total_data.loc[(total_data[col2] >= 60) & (total_data[col2] < 70), col1])
        course_mean.append(total_data.loc[(total_data[col2] >= 70) & (total_data[col2] < 80), col1])
        course_mean.append(total_data.loc[(total_data[col2] >= 80) & (total_data[col2] < 90), col1])
        course_mean.append(total_data.loc[total_data[col2] >= 90, col1])
        plt.boxplot(
            x=course_mean, 
            labels=['0-60','60-70','70-80','80-90','90-100'],
            medianprops={'color':'orange'},
            boxprops=dict(color="blue"),
            whiskerprops = {'color': "black"},
            capprops = {'color': "red"},
            flierprops={'color':'purple','markeredgecolor':"purple"}
        )
        plt.xlabel(col2 + ' score')
        plt.ylabel('course ' + col1)
        plt.title(col2 + ' - course ' + col1)
        plt.show()


# In[8]:


for col in course_col:
    list_male = total_data.loc[total_data['xb'] == 1, col]
    list_female = total_data.loc[total_data['xb'] == 2, col]
    
    plt.hist(list_male, rwidth=0.9, bins=20, range=(0, max(list_male)), label='male')
    plt.hist(list_female, rwidth=0.9, bins=20, range=(0, max(list_male)), label='female')
    plt.xlabel(col)
    plt.ylabel('numbers')
    plt.title('gender-'+col)
    plt.legend()
    plt.show()


# In[22]:


corr_matrix = np.random.rand(len(score_col), len(app_col))
for col1 in app_col:
    for col2 in score_col:
        print('corrcoef ', col1, ' ', col2, ':', np.corrcoef(total_data[col1], total_data[col2])[0, 1])
        corr_matrix[score_col.index(col2), app_col.index(col1)] = np.corrcoef(total_data[col1], total_data[col2])[0, 1]
print(corr_matrix)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr_matrix, interpolation='nearest', cmap='Oranges')
fig.colorbar(cax)
ax.set_xticks(range(len(app_col)))
ax.set_xticklabels([''] + app_col, rotation = 45)
ax.set_yticklabels([''] + score_col)
plt.show()


rf_matrix = np.random.rand(len(score_col), len(app_col))
for col in score_col:
    rf = RandomForestRegressor(n_estimators=20)
    rf.fit(total_data[app_col], total_data[col])
    print(rf.feature_importances_)
    rf_matrix[score_col.index(col)] = rf.feature_importances_
print(rf_matrix)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(rf_matrix, interpolation='nearest', cmap='Oranges')
fig.colorbar(cax)
ax.set_xticks(range(len(app_col)))
ax.set_xticklabels([''] + app_col, rotation = 45)
ax.set_yticklabels([''] + score_col)
plt.show()


# In[23]:


for col1 in app_col:
    for col2 in score_col:
        course_mean = []
        course_mean.append(total_data.loc[total_data[col2] < 60, col1])
        course_mean.append(total_data.loc[(total_data[col2] >= 60) & (total_data[col2] < 70), col1])
        course_mean.append(total_data.loc[(total_data[col2] >= 70) & (total_data[col2] < 80), col1])
        course_mean.append(total_data.loc[(total_data[col2] >= 80) & (total_data[col2] < 90), col1])
        course_mean.append(total_data.loc[total_data[col2] >= 90, col1])
        plt.boxplot(
            x=course_mean, 
            labels=['0-60','60-70','70-80','80-90','90-100'],
            medianprops={'color':'orange'},
            boxprops=dict(color="blue"),
            whiskerprops = {'color': "black"},
            capprops = {'color': "red"},
            flierprops={'color':'purple','markeredgecolor':"purple"}
        )
        plt.xlabel(col2 + ' score')
        plt.ylabel('app_use ' + col1)
        plt.title(col2 + ' - app_use ' + col1)
        plt.show()


# In[24]:


for col in app_col:
    list_male = total_data.loc[total_data['xb'] == 1, col]
    list_female = total_data.loc[total_data['xb'] == 2, col]
    
    plt.hist(list_male, rwidth=0.9, bins=20, range=(0, max(list_male)), label='male')
    plt.hist(list_female, rwidth=0.9, bins=20, range=(0, max(list_male)), label='female')
    plt.xlabel(col)
    plt.ylabel('time')
    plt.title('gender-'+col)
    plt.legend()
    plt.show()


# In[9]:


for col in score_col:
    list_male = total_data.loc[total_data['xb'] == 1, col]
    list_female = total_data.loc[total_data['xb'] == 2, col]
    
    plt.hist(list_male, rwidth=0.9, bins=20, range=(0, max(list_male)), label='male')
    plt.hist(list_female, rwidth=0.9, bins=20, range=(0, max(list_male)), label='female')
    plt.xlabel(col)
    plt.ylabel('numbers')
    plt.title('gender-'+col)
    plt.legend()
    plt.show()


# In[ ]:




