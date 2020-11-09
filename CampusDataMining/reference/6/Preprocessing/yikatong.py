import numpy as np
import pandas as pd
import datetime


ykt2018 = pd.read_csv('../SourceData/ykt_jyrz_2018.txt',sep = ';',usecols = ['XH','JYJE','JYRQ','JYSJ','JYDD'])
ykt2019 = pd.read_csv('../SourceData/ykt_jyrz_2019.txt',sep = ';',usecols = ['XH','JYJE','JYRQ','JYSJ','JYDD'])
ykt = ykt2018.append(ykt2019).reset_index(drop=True)
ykt2019.columns = ['xh','jyje','jyrq','jysj','jydd']
ykt2018.columns = ['xh','jyje','jyrq','jysj','jydd']
ykt.columns = ['xh','jyje','jyrq','jysj','jydd']
begin = datetime.datetime(2018,1,1)
ykt['jyrq'] = pd.to_datetime(ykt['jyrq'],format='%Y-%m-%d')
cjxx1 = pd.read_csv('../SourceData/bks_cjxx_out1-1.csv',usecols = ['xh'])
cjxx2 = pd.read_csv('../SourceData/bks_cjxx_out1-2.csv',usecols = ['xh'])
cjxx = cjxx1.append(cjxx2)
xjjbsjxx = pd.read_csv('../SourceData/bks_xjjbsjxx_out1.csv',usecols = ['xh'])
xsjbsjxx = pd.read_csv('../SourceData/bks_xsjbsjxx_out.csv',usecols = ['xh'])
set_cjxx = set(cjxx['xh'])
set_xjjbsjxx = set(xjjbsjxx['xh'])
set_xsjbsjxx = set(xsjbsjxx['xh'])

set_xh = set_cjxx.intersection(set_xsjbsjxx)
set_xh = set_xh.intersection(set_xjjbsjxx)
counter=0
print(1111111111)
average = {}
for name in set_xh:
    average[name] = [0]*71

print(len(list(average.keys())))

counter =0
for index in ykt.index:
    if(str(ykt.at[index,'xh'])[0].isalpha() == True):
        continue
    if(int(ykt.at[index,'xh']) not in set_xh):
        continue
    else:
        counter+=1
        if(counter%10000==0):
            print(counter)
        average[int(ykt.at[index,'xh'])][int((ykt.at[index,'jyrq'] - begin).days/7)] +=ykt.at[index,'jyje']

print( average)
df = pd.DataFrame(average)
df.to_csv('../Data/average1.csv',index = None, encoding='utf-8')


