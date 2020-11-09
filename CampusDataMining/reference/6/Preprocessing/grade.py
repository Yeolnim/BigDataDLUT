import numpy as np
import pandas as pd

score = {}
cjxx1 = pd.read_csv('../SourceData/bks_cjxx_out1-1.csv',usecols = ['xh','xn','xqm','ksrq','kch','kxh','kccj','xf','kcsxdm','xdfsdm'])
cjxx2 = pd.read_csv('../SourceData/bks_cjxx_out1-2.csv',usecols = ['xh','xn','xqm','ksrq','kch','kxh','kccj','xf','kcsxdm','xdfsdm'])
cjxx = cjxx1.append(cjxx2).reset_index(drop=True)
data = cjxx[['xh','kcsxdm','xf','kccj','kch']]
print(cjxx.shape)
'''for index in data.index:
    if(data.at[index,'kcsxdm'] == 1.0):
        if(data.loc[index, 'xh'] not in score_bixiu.keys()):
            score_bixiu[data.loc[index, 'xh']] = [data.loc[index,'xf']*data.loc[index,'kccj'],1]
        else:
            score_bixiu[data.at[index, 'xh']][0] += (data.at[index,'xf'])*(data.at[index,'kccj'])
            score_bixiu[data.at[index, 'xh']][1] += 1
    if (data.at[index, 'kcsxdm'] == 0.0 or data.at[index, 'kcsxdm'] == 3.0):
        if (data.loc[index, 'xh'] not in score_xuanxiu.keys()):
            score_xuanxiu[data.loc[index, 'xh']] = [data.loc[index, 'xf'] * data.loc[index, 'kccj'], 1]
        else:
            score_xuanxiu[data.at[index, 'xh']][0] += (data.at[index, 'xf']) * (data.at[index, 'kccj'])
            score_xuanxiu[data.at[index, 'xh']][1] += 1
    if (data.at[index, 'kcsxdm'] == 1.0):
        if(data.at[index,'xf'] >=3):
            if (data.loc[index, 'xh'] not in score_bixiu1.keys()):
                score_bixiu1[data.loc[index, 'xh']] = [data.loc[index, 'xf'] * data.loc[index, 'kccj'], 1]
            else:
                score_bixiu1[data.at[index, 'xh']][0] += (data.at[index, 'xf']) * (data.at[index, 'kccj'])
                score_bixiu1[data.at[index, 'xh']][1] += 1
        else:
            if (data.loc[index, 'xh'] not in score_bixiu2.keys()):
                score_bixiu2[data.loc[index, 'xh']] = [data.loc[index, 'xf'] * data.loc[index, 'kccj'], 1]
            else:
                score_bixiu2[data.at[index, 'xh']][0] += (data.at[index, 'xf']) * (data.at[index, 'kccj'])
                score_bixiu2[data.at[index, 'xh']][1] += 1'''



for index in data.index:
    if (data.loc[index, 'xh'] not in score.keys()):
        score[data.loc[index, 'xh']] = [0,0,0,0,0,0,0,0]
    if(data.at[index,'kcsxdm'] == 1.0):
        score[data.at[index, 'xh']][0] += (data.at[index,'xf'])*(data.at[index,'kccj'])
        score[data.at[index, 'xh']][1] += data.at[index, 'xf']
        if (data.at[index, 'xf'] >= 3):
            score[data.at[index, 'xh']][4] += (data.at[index, 'xf']) * (data.at[index, 'kccj'])
            score[data.at[index, 'xh']][5] += data.at[index, 'xf']
        else:
            score[data.at[index, 'xh']][6] += (data.at[index, 'xf']) * (data.at[index, 'kccj'])
            score[data.at[index, 'xh']][7] += data.at[index, 'xf']
    if (data.at[index, 'kcsxdm'] == 0.0 or data.at[index, 'kcsxdm'] == 3.0):
        score[data.at[index, 'xh']][2] += (data.at[index, 'xf']) * (data.at[index, 'kccj'])
        score[data.at[index, 'xh']][3] += data.at[index, 'xf']

res_dict={}
set_key = score.keys()
result = pd.DataFrame(columns = ['xh','bx','xx','dbx','xbx'])
for key in set_key:
    res_dict[key] = [0,0,0,0]
    if(score[key][1]==0):
        res_dict[key][0] = 0
    else:
        res_dict[key][0] = score[key][0]/score[key][1]
    if (score[key][3] == 0):
        res_dict[key][1] = 0
    else:
        res_dict[key][1] = score[key][2] / score[key][3]
    if (score[key][5] == 0):
        res_dict[key][2] = 0
    else:
        res_dict[key][2] = score[key][4] / score[key][5]
    if (score[key][7] == 0):
        res_dict[key][3] = 0
    else:
        res_dict[key][3] = score[key][6] / score[key][7]
for key in set_key:
    result = result.append(pd.DataFrame({'xh':[key], 'bx':[res_dict[key][0]], 'xx':[res_dict[key][1]], 'dbx':[res_dict[key][2]],'xbx':[res_dict[key][3]]}), ignore_index=True)

result.to_csv('../Data/score.csv',index=None, encoding='utf-8')
print(result.head())