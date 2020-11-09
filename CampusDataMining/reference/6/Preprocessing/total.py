import numpy as np
import pandas as pd

score = pd.read_csv('../Data/score.csv')
cost = pd.read_csv('../Data/吃饭花费1.csv',engine='python')
chifan = pd.read_csv('../Data/吃饭花费.csv',engine='python').drop(['sum'],axis=1)
junfen = pd.read_csv('../Data/学期均分.csv',engine='python')
fail = pd.read_csv('../Data/学期挂科.csv',engine='python')
week = pd.read_csv('../Data/mean.csv')
zaoshang = pd.read_csv('../Data/早上和下午第一节.csv',engine='python')
xueye = pd.read_csv('../Data/xueyejinggao.csv')
app = pd.read_csv('../Data/app_use.csv')
xsjbsjxx = pd.read_csv('../SourceData/bks_xsjbsjxx_out.csv',usecols = ['xh','xb'])


re = pd.merge(score,cost,on='xh')
re = pd.merge(re,xsjbsjxx,on = 'xh')
re = pd.merge(re,xueye,on = 'xh')
re = pd.merge(re, week, on = 'xh')
re = pd.merge(re, zaoshang, on = 'xh')
re = pd.merge(re,chifan,on='xh')
re = pd.merge(re,fail,on='xh')
re = pd.merge(re, junfen ,on= 'xh')
re = pd.merge(re, app, on='xh')
print(re.shape)

re = re.replace('男',1)
re = re.replace('女',2)
re = re.replace('未知性别',3)


re.to_csv('../Data/total1.csv',index=None, encoding='utf-8')

