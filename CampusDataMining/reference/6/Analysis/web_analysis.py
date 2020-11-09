import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

print(set_xh)
print(len(set_xh))
print(type(set_xh))

app_data = pd.read_csv('../SourceData/t_app_online_out.txt',encoding='utf-8', sep=';', engine='python')
print(app_data.head())
print(app_data.columns)

app = []
for index in app_data.index:
    app.append(app_data.at[index, 'appname'])

app = (np.unique(app)).tolist()
print(app)

app_use_time = np.array(np.zeros( len(app) ))
for index in app_data.index:
    app_use_time[app.index(app_data.at[index, 'appname'])] += 1


dic = {}
for item in app:
    dic[item] = int(app_use_time[app.index(item)])
print(dic)

for key in list(dic.keys()):
    if dic[key] < 15000:
        del dic[key]
print(dic)


app_watch_times = sorted(zip(dic.values(), dic.keys()))
print(app_watch_times)

x = range(len(app_watch_times))
app_watch_times = dict(app_watch_times)
print(app_watch_times.keys())
print(app_watch_times.values())

plt.rcParams['font.sans-serif']=['SimHei']

plt.bar(x, app_watch_times.keys(), color='orange')
plt.xticks(x, app_watch_times.values(), rotation = 45)
plt.show()


zhibo = ['YY直播', '一直播','企鹅电竞', '全民直播', '战旗直播','斗鱼直播', '映客直播', '来疯直播',
         '熊猫直播','第一视频', '网易直播', '花椒直播', '虎牙直播', '龙珠直播', '腾讯NOW', '触手tv', '网易CC']
duanshipin = ['PP视频','哔哩哔哩', '快手', '抖音', '梨视频', '火山小视频',  '看看视频', '爱拍原创','腾讯微视']
shipin = [ '乐视', '今视网', '优酷', '央视影音', '搜狐视频', '江苏TV', '芒果TV']
youxi = ['QQ炫舞', 'QQ炫舞手游', 'QQ飞车', 'QQ飞车手游',
     '全民突击', '剑灵',  '土豆', '地下城', '天天酷跑',
     '暴风影音', '梦三国', '欢乐升级', '欢乐斗地主',
     '火影忍者', '爱奇艺', '王者荣耀', '疾风之刃', '穿越火线',
     '穿越火线：枪战王者',  '第五人格', '绝地求生', '绝地求生：全军出击',
      '英雄联盟', '逆战', '阴阳师', '雷霆战机',
     '风行', '魔兽', '龙之谷']
tengxun = ['腾讯']

use_times = [0] * 5
for item in app:
    if item in zhibo:
        use_times[0] += int(app_use_time[app.index(item)])
    elif item in duanshipin:
        use_times[1] += int(app_use_time[app.index(item)])
    elif item in shipin:
        use_times[2] += int(app_use_time[app.index(item)])
    elif item in youxi:
        use_times[3] += int(app_use_time[app.index(item)])
    elif item in tengxun:
        use_times[4] += int(app_use_time[app.index(item)])
print(use_times)

labels = ['直播', '短视频', '视频', '游戏', '即时通讯']
explode = (0,0,0.08,0.16,0)
plt.pie(use_times, explode=explode,labels=labels, autopct='%1.1f%%', shadow=False,
        textprops = {'fontsize':16, 'color':'black'})
plt.title('app使用占比')
plt.show()


import datetime

app_use_period = np.array(np.zeros(len(app)))
for index in app_data.index:
    start = app_data.at[index, 'starttime'][:19]
    start_time = datetime.datetime.strptime(start, '%Y/%m/%d %H:%M:%S')

    end = app_data.at[index, 'lasttime'][:19]
    end_time = datetime.datetime.strptime(end, '%Y/%m/%d %H:%M:%S')

    app_use_period[app.index(app_data.at[index, 'appname'])] += (end_time - start_time).seconds / 60.0

print(app_use_period)

dic1 = {}
for item in app:
    dic1[item] = int(app_use_period[app.index(item)])
print(dic1)

for key in list(dic1.keys()):
    if dic1[key] < 300000:
        del dic1[key]
print(dic1)

app_watch_period = sorted(zip(dic1.values(), dic1.keys()))
print(app_watch_period)

x = range(len(app_watch_period))
app_watch_period = dict(app_watch_period)
print(app_watch_period.keys())
print(app_watch_period.values())

plt.rcParams['font.sans-serif']=['SimHei']

plt.bar(x, app_watch_period.keys(), color='orange')
plt.xticks(x, app_watch_period.values(), rotation = 45)
plt.show()

use_period = [0] * 5
for item in app:
    if item in zhibo:
        use_period[0] += app_use_period[app.index(item)]
    elif item in duanshipin:
        use_period[1] += app_use_period[app.index(item)]
    elif item in shipin:
        use_period[2] += app_use_period[app.index(item)]
    elif item in youxi:
        use_period[3] += app_use_period[app.index(item)]
    elif item in tengxun:
        use_period[4] += app_use_period[app.index(item)]
print(use_period)

labels = ['直播', '短视频', '视频', '游戏', '即时通讯']
explode = (0,0,0.08,0.16,0)
plt.pie(use_period, explode=explode,labels=labels, autopct='%1.1f%%', shadow=False, textprops = {'fontsize':12, 'color':'black'})
plt.title('app使用时长占比')
plt.show()
