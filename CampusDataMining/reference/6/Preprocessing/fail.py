import numpy as np
import pandas as pd

cjxx1 = pd.read_csv('../SourceData/bks_cjxx_out1-1.csv',usecols = ['xh','xn','xqm','ksrq','kch','kxh','kccj','xf','kcsxdm','xdfsdm'])
cjxx2 = pd.read_csv('../SourceData/bks_cjxx_out1-2.csv',usecols = ['xh','xn','xqm','ksrq','kch','kxh','kccj','xf','kcsxdm','xdfsdm'])
cjxx = cjxx1.append(cjxx2).reset_index(drop=True)

cjxx = cjxx.astype({'xn':'str'})
dict_fail={}
dict_pass = {}
dict_score = {}
dict_res={}

print(00000)
counter=0
'''for index in cjxx.index:
    if(cjxx.at[index,'xh'] == 201437059 and (cjxx.at[index,'kcsxdm'] == 1 or cjxx.at[index, 'kcsxdm'] == 3 or cjxx.at[index, 'kcsxdm'] == 0) and cjxx.at[index,'kccj']>=60):
        print(cjxx.at[index,'xn'],cjxx.at[index,'xqm'],cjxx.at[index,'kch'],cjxx.at[index,'xf'])'''
for index in cjxx.index:
    counter+=1
    if(counter%10000==0):
        print(counter)
    if(cjxx.at[index,'xh'] <201400000 or cjxx.at[index,'xh']> 201900000):
        continue
    if(cjxx.at[index,'xh'] not in dict_score.keys()):
        dict_score[cjxx.at[index, 'xh']] = [0] *100 
        dict_fail[cjxx.at[index,'xh']] = [0] * 100
        dict_pass[cjxx.at[index, 'xh']] = [0] * 20
    if(cjxx.at[index,'kcsxdm'] == 1):
        start, end = cjxx.at[index, 'xn'].split('-')
        if(int(start)<2014 or int(start) >=2019):
            continue
        dict_score[cjxx.at[index, 'xh']][2 * (3 * int(start) + cjxx.at[index, 'xqm'] - 6043) + 1] += cjxx.at[index, 'xf']
        dict_score[cjxx.at[index, 'xh']][2 * (3 * int(start) + cjxx.at[index, 'xqm'] - 6043)] += cjxx.at[index, 'xf']*cjxx.at[index,'kccj']
        if (cjxx.at[index, 'kccj'] >= 60):
            dict_pass[cjxx.at[index, 'xh']][3 * int(start) + cjxx.at[index, 'xqm'] - 6043] += cjxx.at[index, 'xf']
        else:
            dict_fail[cjxx.at[index, 'xh']][3 * int(start) + cjxx.at[index, 'xqm'] - 6043] += 1
    if (cjxx.at[index, 'kcsxdm'] == 3 or cjxx.at[index, 'kcsxdm'] == 0):
        start, end = cjxx.at[index, 'xn'].split('-')
        if (int(start) < 2014 or int(start) >= 2019):
            continue
        if (cjxx.at[index, 'kccj'] >= 60):
            dict_pass[cjxx.at[index, 'xh']][3 * int(start) + cjxx.at[index, 'xqm'] - 6043] += cjxx.at[index, 'xf']


print(1111111)
counter=0
for key in dict_score.keys():
    counter += 1
    if (counter % 1000 == 0):
        print(counter)
    dict_res[key] = [0]*13
    for i in range(13):
        if(dict_score[key][2*i+1] != 0):
            dict_res[key][i] = dict_score[key][2*i]/dict_score[key][2*i+1]
        else:
            dict_res[key][i] = 0


print(222222)
counter=0
sum1=0
sum2=0
sum3=0
result1 = pd.DataFrame(columns = ['xh','xueyejinggao1','xueyejinggao2'])
for key in dict_pass.keys():
    counter += 1
    if (counter % 1000 == 0):
        print(counter)
    temp = int(int(key)/100000)%10
    if(temp == 4):
       if(dict_pass[key][0] >= 15 and (dict_pass[key][1] + dict_pass[key][2]) >= 15 and dict_pass[key][3] >= 15 and (dict_pass[key][4] + dict_pass[key][5]) >= 15
           and dict_pass[key][6] >= 15 and (dict_pass[key][7] + dict_pass[key][8]) >= 15):
           result1 = result1.append(pd.DataFrame({'xh':[key],'xueyejinggao1':1}),ignore_index=True)
       else:
           result1 = result1.append(pd.DataFrame({'xh': [key], 'xueyejinggao1': 2}),ignore_index=True)
    if (temp == 5):
        if (dict_pass[key][3] >= 15 and (dict_pass[key][4] + dict_pass[key][5]) >= 15 and dict_pass[key][6] >= 15 and (dict_pass[key][7] + dict_pass[key][8]) >= 15 and dict_pass[key][9] >= 15 and (dict_pass[key][10] + dict_pass[key][11]) >= 15):
            result1 = result1.append(pd.DataFrame({'xh': [key], 'xueyejinggao1': 1}),ignore_index=True)
        else:
            result1 = result1.append(pd.DataFrame({'xh': [key], 'xueyejinggao1': 2}),ignore_index=True)
    if (temp == 6):
        if (dict_pass[key][6] >= 15 and (dict_pass[key][7] + dict_pass[key][8]) >= 15 and dict_pass[key][9] >= 15 and (dict_pass[key][10] + dict_pass[key][11]) >= 15 and dict_pass[key][12] >= 15):
            result1 = result1.append(pd.DataFrame({'xh': [key], 'xueyejinggao1': 1}),ignore_index=True)
        else:
            result1 = result1.append(pd.DataFrame({'xh': [key], 'xueyejinggao1': 2}),ignore_index=True)
    if(temp == 7):
        if (dict_pass[key][9] >= 15 and (dict_pass[key][10] + dict_pass[key][11]) >= 15 and dict_pass[key][12] >= 15):
            result1 = result1.append(pd.DataFrame({'xh': [key], 'xueyejinggao2': 1}), ignore_index=True)
        else:
            result1 = result1.append(pd.DataFrame({'xh': [key], 'xueyejinggao2': 2}), ignore_index=True)
    if(temp == 8):
        if(dict_pass[key][12] >= 15):
            sum1+=1
        else:
            sum3+=1
        sum2+=1
        result1 = result1.append(pd.DataFrame({'xh': [key]}), ignore_index=True)
#df = pd.DataFrame(dict_pass)
#df.to_csv('暂时.csv')
print(sum1,sum2,sum3)
result1.to_csv('../Data/xueyejinggao.csv',index=None,encoding='utf-8')
result = pd.DataFrame(columns = ['xh','2014-1','2014-2','2014-3','2015-1','2015-2','2015-3','2016-1','2016-2','2016-3','2017-1','2017-2','2017-3','2018-1'])
fail = pd.DataFrame(columns = ['xh','2014-1','2014-2','2014-3','2015-1','2015-2','2015-3','2016-1','2016-2','2016-3','2017-1','2017-2','2017-3','2018-1'])
counter=0
for key in dict_res.keys():
    counter += 1
    if (counter % 1000 == 0):
        print(counter)
    temp = int(key/100000)%10
    if(temp == 4):
        result = result.append(pd.DataFrame(
            {'xh': [key], '2014-1':dict_res[key][0], '2014-2':dict_res[key][1], '2014-3':dict_res[key][2], '2015-1':dict_res[key][3], '2015-2':dict_res[key][4], '2015-3':dict_res[key][5], '2016-1':dict_res[key][6], '2016-2':dict_res[key][7],'2016-3':dict_res[key][8], '2017-1':dict_res[key][9],'2017-2':dict_res[key][10], '2017-3':dict_res[key][11], '2018-1':dict_res[key][12]}), ignore_index=True)
    if (temp == 5):
        result = result.append(pd.DataFrame(
            {'xh': [key],
             '2015-1': dict_res[key][3], '2015-2': dict_res[key][4], '2015-3': dict_res[key][5],
             '2016-1': dict_res[key][6], '2016-2': dict_res[key][7], '2016-3': dict_res[key][8],
             '2017-1': dict_res[key][9], '2017-2': dict_res[key][10], '2017-3': dict_res[key][11],
             '2018-1': dict_res[key][12]}), ignore_index=True)
    if (temp == 6):
        result = result.append(pd.DataFrame(
            {'xh': [key],
             '2016-1': dict_res[key][6], '2016-2': dict_res[key][7], '2016-3': dict_res[key][8],
             '2017-1': dict_res[key][9], '2017-2': dict_res[key][10], '2017-3': dict_res[key][11],
             '2018-1': dict_res[key][12]}), ignore_index=True)
    if (temp == 7):
        result = result.append(pd.DataFrame(
            {'xh': [key],
             '2017-1': dict_res[key][9], '2017-2': dict_res[key][10], '2017-3': dict_res[key][11],
             '2018-1': dict_res[key][12]}), ignore_index=True)
    if (temp == 8):
        result = result.append(pd.DataFrame(
            {'xh': [key],
             '2018-1': dict_res[key][12]}), ignore_index=True)

counter=0
for key in dict_fail.keys():
    counter += 1
    if (counter % 1000 == 0):
        print(counter)
    temp = int(key/100000)%10
    if (temp == 4):
        fail = fail.append(pd.DataFrame(
            {'xh': [key], '2014-1': dict_fail[key][0], '2014-2': dict_fail[key][1], '2014-3': dict_fail[key][2],
             '2015-1': dict_fail[key][3], '2015-2': dict_fail[key][4], '2015-3': dict_fail[key][5],
             '2016-1': dict_fail[key][6], '2016-2': dict_fail[key][7], '2016-3': dict_fail[key][8],
             '2017-1': dict_fail[key][9], '2017-2': dict_fail[key][10], '2017-3': dict_fail[key][11],
             '2018-1': dict_fail[key][12]}), ignore_index=True)
    if (temp == 5):
        fail = fail.append(pd.DataFrame(
            {'xh': [key],
             '2015-1': dict_fail[key][3], '2015-2': dict_fail[key][4], '2015-3': dict_fail[key][5],
             '2016-1': dict_fail[key][6], '2016-2': dict_fail[key][7], '2016-3': dict_fail[key][8],
             '2017-1': dict_fail[key][9], '2017-2': dict_fail[key][10], '2017-3': dict_fail[key][11],
             '2018-1': dict_fail[key][12]}), ignore_index=True)
    if (temp == 6):
        fail = fail.append(pd.DataFrame(
            {'xh': [key],
             '2016-1': dict_fail[key][6], '2016-2': dict_fail[key][7], '2016-3': dict_fail[key][8],
             '2017-1': dict_fail[key][9], '2017-2': dict_fail[key][10], '2017-3': dict_fail[key][11],
             '2018-1': dict_fail[key][12]}), ignore_index=True)
    if (temp == 7):
        fail = fail.append(pd.DataFrame(
            {'xh': [key],
             '2017-1': dict_fail[key][9], '2017-2': dict_fail[key][10], '2017-3': dict_fail[key][11],
             '2018-1': dict_fail[key][12]}), ignore_index=True)
    if (temp == 8):
        fail = fail.append(pd.DataFrame(
            {'xh': [key],
             '2018-1': dict_fail[key][12]}), ignore_index=True)

print(fail.head())
print(result.head())

fail.to_csv('../Data/学期挂科.csv',index=None,encoding='utf-8')
result.to_csv('../Data/学期均分.csv',index=None,encoding='utf-8')