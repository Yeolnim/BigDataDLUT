import numpy as np
import pandas as pd

#cjxx1 = pd.read_csv('bks_cjxx_out1-1.csv',usecols = ['xh','xn','xqm','ksrq','kch','kxh','kccj','xf','kcsxdm','xdfsdm'])
#cjxx2 = pd.read_csv('bks_cjxx_out1-2.csv',usecols = ['xh','xn','xqm','ksrq','kch','kxh','kccj','xf','kcsxdm','xdfsdm'])
#cjxx = cjxx1.append(cjxx2).reset_index(drop=True)
chifan = pd.read_csv('../Data/吃饭花费.csv',engine='python')
result = pd.DataFrame(columns = ['xh','chifan'])
for index in chifan.index:
    if(chifan.at[index,'time'] == 0):
        result = result.append(pd.DataFrame(
            {'xh': [chifan.at[index, 'xh']], 'chifan': 0}),
            ignore_index=True)
    else:
        result = result.append(pd.DataFrame(
            {'xh': [chifan.at[index,'xh']], 'chifan':[chifan.at[index,'sum']/chifan.at[index,'time']]}), ignore_index=True)
result.to_csv('../Data/吃饭花费1.csv',index=None,encoding='utf-8')
'''pksjxx = pd.read_csv('bks_pksjxx_out.csv',usecols = ['xn','xqm','jxdd','xdrs','zxs','kch','kxh','skxq','skjc','cxjc','zcsm'])
xjjbsjxx = pd.read_csv('bks_xjjbsjxx_out1.csv',usecols = ['xh','yxsh','zym','sznj','xqh'])
xsjbsjxx = pd.read_csv('bks_xsjbsjxx_out.csv',usecols = ['xh','xb'])
set_cjxx = set(cjxx['xh'])
set_xjjbsjxx = set(xjjbsjxx['xh'])
set_xsjbsjxx = set(xsjbsjxx['xh'])

set_xh = set_cjxx.intersection(set_xsjbsjxx)
set_xh = set_xh.intersection(set_xjjbsjxx)
lesson= {}
print(111111)
for name in set_xh:
    lesson[name] = [0]*2

re = pd.merge(cjxx,pksjxx,'inner',on = ['xn','xqm','kch'])
data = re[['xh','skjc']]'''
#print(cjxx.shape)