import numpy as np
import pandas as pd

cjxx1 = pd.read_csv('../SourceData/bks_cjxx_out1-1.csv',usecols = ['xh','xn','xqm','ksrq','kch','kxh','kccj','xf','kcsxdm','xdfsdm'])
cjxx2 = pd.read_csv('../SourceData/bks_cjxx_out1-2.csv',usecols = ['xh','xn','xqm','ksrq','kch','kxh','kccj','xf','kcsxdm','xdfsdm'])
cjxx = cjxx1.append(cjxx2).reset_index(drop=True)
cjxx = cjxx[['xh','xn','xqm','kch']]
#pksjxx = pd.read_csv('bks_pksjxx_out.csv',usecols = ['xn','xqm','jxdd','xdrs','zxs','kch','kxh','skxq','skjc','cxjc','zcsm'])
pksjxx = pd.read_csv('../SourceData/bks_pksjxx_out.csv',usecols = ['xn','xqm','kch','skjc'])
xjjbsjxx = pd.read_csv('../SourceData/bks_xjjbsjxx_out1.csv',usecols = ['xh','yxsh','zym','sznj','xqh'])
xsjbsjxx = pd.read_csv('../SourceData/bks_xsjbsjxx_out.csv',usecols = ['xh','xb'])
set_cjxx = set(cjxx['xh'])
set_xjjbsjxx = set(xjjbsjxx['xh'])
set_xsjbsjxx = set(xsjbsjxx['xh'])

set_xh = set_cjxx.intersection(set_xsjbsjxx)
set_xh = set_xh.intersection(set_xjjbsjxx)
lesson= {}
print(111111)
for name in set_xh:
    lesson[name] = [0]*2
xkxx={}
counter=0
for index in pksjxx.index:
    if((pksjxx.at[index,'xn'],pksjxx.at[index,'xqm'],pksjxx.at[index,'kch']) not in xkxx.keys()):
        counter += 1
        if(counter%10000==0):
            print(counter)
        xkxx[(pksjxx.at[index,'xn'],pksjxx.at[index,'xqm'],pksjxx.at[index,'kch'])] = pksjxx.at[index,'skjc']

for index in cjxx.index:
    if(cjxx.at[index, 'xh'] not in lesson.keys()):
        continue
    if((cjxx.at[index,'xn'],cjxx.at[index,'xqm'],cjxx.at[index,'kch']) in xkxx.keys()):
        counter+=1
        if(counter%10000==0):
            print(counter)
        if(xkxx[(cjxx.at[index,'xn'],cjxx.at[index,'xqm'],cjxx.at[index,'kch'])] == 1):
            lesson[cjxx.at[index, 'xh']][0] += 1
            continue
        if(xkxx[(cjxx.at[index,'xn'],cjxx.at[index,'xqm'],cjxx.at[index,'kch'])] == 5):
            lesson[cjxx.at[index,'xh']][1] += 1
            continue


'''re = pd.merge(cjxx,pksjxx,'inner',on = ['xn','xqm','kch'])
data = re[['xh','skjc']]
counter = 0
for index in data.index:
    if (str(data.at[index, 'xh'])[0].isalpha() == True):
        continue
    if (int(data.at[index, 'xh']) not in set_xh):
        continue
    else:
        counter+=1
        if(counter%1000==0):
            print(counter)
        if(int(data.at[index,'skjc']) == 1):
            lesson[data.at[index,'xh']][0]+=1
        if (int(data.at[index, 'skjc']) == 5):
            lesson[data.at[index, 'xh']][1] += 1'''

df = pd.DataFrame(columns = ['xh','morning','afternoon'])
for key in list(lesson.keys()):
    df = df.append(pd.DataFrame({'xh':[key],'morning':[lesson[key][0]],'afternoon':[lesson[key][1]]}))
df.to_csv('../Data/早上和下午第一节.csv',index=None,encoding='utf-8')