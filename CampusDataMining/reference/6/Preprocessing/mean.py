import numpy as np
import pandas as pd

dict2={}
df = pd.read_csv('../Data/average1.csv')
dict1 = {col:df[col].tolist() for col in df.columns}
temp = []
for key in list(dict1.keys()):
    if(key not in dict2.keys()):
        dict2[key] = [0]*2
    if(int(int(key)/100000)%10  == 8):
        temp = dict1[key][34:54]+ dict1[key][59:70]
        temp_mean = np.mean(temp)
        temp_var =  np.var(temp)
        dict2[key][0] = temp_mean
        dict2[key][1] = temp_var
    else:
        temp = dict1[key][0:2]+dict1[key][9:29]+dict1[key][34:54]+dict1[key][59:70]
        temp_mean = np.mean(temp)
        temp_var = np.var(temp)
        dict2[key][0] = temp_mean
        dict2[key][1] = temp_var

result = pd.DataFrame(columns = ['xh','mean','var'])
for key in list(dict2.keys()):
    result = result.append(pd.DataFrame({'xh':[key],'mean':[dict2[key][0]],'var':[dict2[key][1]]}),ignore_index=True)
result.to_csv('../Data/mean.csv',index=None,encoding='utf-8')
print(result.head())