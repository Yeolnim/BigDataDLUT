#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

total_data = pd.read_csv('../Data/total1.csv', encoding='utf-8')
print(total_data.head())
print(total_data.columns)


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
#get_ipython().run_line_magic('matplotlib', 'inline')

cosume_col = ['mean', 'var', 'chifan', 'time',
              'duanshipin_time', 'shipin_time', 'zhibo_time',
               'youxi_time', 'tengxun_time', 'duanshipin_period', 'shipin_period',
               'zhibo_period', 'youxi_period', 'tengxun_period']
cosume_list = []
for index in total_data.index:
    cosume = np.random.rand(len(cosume_col))
    for col in cosume_col:
        cosume[cosume_col.index(col)] = total_data.at[index, col]
    cosume_list.append(cosume)

print(type(cosume_list))
print(type(np.array(cosume_list)))
print(np.array(cosume_list).shape)
cosume_array = np.array(cosume_list)


# In[3]:


from sklearn.preprocessing import scale
print(cosume_array)
cosume_array[:, 0] = cosume_array[:, 0]/ cosume_array[:, 0].mean()
cosume_array[:, 1] = cosume_array[:, 1]/ cosume_array[:, 1].mean()
X = scale(cosume_array)
print(X)


# In[10]:


y_pred = DBSCAN(eps=1.6).fit_predict(X)
print('finish DBSCAN')
print(min(y_pred))
print(max(y_pred))

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

color = ['purple', 'orange', 'red', 'green', 'pink', 'black', 'blue', 'yellow']

color_pred = []
for i in range(len(y_pred)):
    color_pred.append(color[y_pred[i] + 1])
    
pca = PCA(n_components=2)
X_2D_PCA = pca.fit_transform(X)
plt.scatter(X_2D_PCA[:, 0], X_2D_PCA[:, 1], c=color_pred, s=6, alpha=0.6)
plt.title('PCA')
plt.show()

tsne = TSNE(n_components=2, learning_rate=100)
X_2D = tsne.fit_transform(X)
plt.scatter(X_2D[:, 0], X_2D[:, 1], c=color_pred, s=6, alpha=0.6)
plt.title('t-SNE')
plt.show()


# In[ ]:




