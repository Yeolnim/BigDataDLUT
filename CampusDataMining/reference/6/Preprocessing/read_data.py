import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
myfont=FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf',size=14)
sns.set(font=myfont.get_name())
data = pd.read_csv('C:\\Users\\12080\Desktop\\bigdata\\data2\\bks_pksjxx_out.csv',encoding='utf-8')
#plt.hist(data['skjc'], color=sns.desaturate("indianred", .8))
x=[1,2,3,4,5,6,7,8,9,10]
x.append('异常值')
y=[0]*11
for index in data.index:
    if(data.at[index,'skjc']>10):
        y[10]+=1
    else:
        j = data.at[index,'skjc']
        y[j-1]+=1
plt.bar(range(len(y)), y,tick_label = x)
for a,b in zip(x,y):
    plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=7)