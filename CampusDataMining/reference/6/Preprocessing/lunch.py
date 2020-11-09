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
#end =datetime.datetime(2019,1,12)
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
    average[name] = [0]*2
sum1=0
sum4=0
counter =0
sum5=0
for index in ykt.index:
    if(str(ykt.at[index,'xh'])[0].isalpha() == True):
        sum4+=1
        continue
    if(int(ykt.at[index,'xh']) not in set_xh):
        sum5+=1
        continue
    else:
        if('食堂'in str(ykt.at[index,'jydd']) and int(ykt.at[index,'jyje']) > 500):
            sum1+=1
            average[int(ykt.at[index,'xh'])][0]+=ykt.at[index,'jyje']
            average[int(ykt.at[index, 'xh'])][1] += 1

            print(average[int(ykt.at[index, 'xh'])][0], average[int(ykt.at[index, 'xh'])][1])

            counter += 1
            if(counter % 10000 == 0):
                print(counter)

'''print(sum1)
sum2=0
sum3=0
for key in list(average.keys()):
    if(average[key][1] == 0):
        continue
        #print(key,average[key][1])
    else:
        print(key,average[key][1])

print(sum2,sum3)
print(sum4,sum5)'''
df = pd.DataFrame(columns = ['xh','sum','time'])
for key in list(average.keys()):
    df = df.append(pd.DataFrame({'xh':[key],'sum':[average[key][0]],'time':[average[key][1]]}))
df.to_csv('../Data/吃饭花费.csv',index=None,encoding='utf-8')