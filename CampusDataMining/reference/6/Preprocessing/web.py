import pandas as pd
import datetime


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


columns = ['xh', 'duanshipin_time', 'shipin_time', 'zhibo_time', 'youxi_time', 'tengxun_time',
           'duanshipin_period', 'shipin_period', 'zhibo_period', 'youxi_period', 'tengxun_period']

app_data = pd.DataFrame(columns=columns)
app_data['xh'] = pd.Series(list(set_xh))
app_data = app_data.fillna(0)


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

zhibo_time = dict.fromkeys(list(set_xh), 0)
duanshipin_time = dict.fromkeys(list(set_xh), 0)
shipin_time = dict.fromkeys(list(set_xh), 0)
youxi_time = dict.fromkeys(list(set_xh), 0)
tengxun_time = dict.fromkeys(list(set_xh), 0)

zhibo_period = dict.fromkeys(list(set_xh), 0)
duanshipin_period = dict.fromkeys(list(set_xh), 0)
shipin_period = dict.fromkeys(list(set_xh), 0)
youxi_period = dict.fromkeys(list(set_xh), 0)
tengxun_period = dict.fromkeys(list(set_xh), 0)

app = pd.read_csv('../SourceData/t_app_online_out.txt',encoding='utf-8', sep=';', engine='python')
print(app.head())
print(app.columns)

for index in app.index:
    if index % 10000 == 0:
        print(index)

    xh = app.at[index, 'xh']
    if xh[0].isalpha() == True:
        continue
    if xh[2] == '.' or xh[3] == '.':
        continue
    xh = int(xh)

    if xh in set_xh:
        appname = app.at[index, 'appname']

        start = app.at[index, 'starttime'][:19]
        start_time = datetime.datetime.strptime(start, '%Y/%m/%d %H:%M:%S')
        end = app.at[index, 'lasttime'][:19]
        end_time = datetime.datetime.strptime(end, '%Y/%m/%d %H:%M:%S')
        period = (end_time - start_time).seconds / 60.0

        if appname in zhibo:
            zhibo_time[xh] += 1
            zhibo_period[xh] += period

        elif appname in duanshipin:
            duanshipin_time[xh] += 1
            duanshipin_period[xh] += period

        elif appname in shipin:
            shipin_time[xh] += 1
            shipin_period[xh] += period

        elif appname in youxi:
            youxi_time[xh] += 1
            youxi_period[xh] += period

        elif appname in tengxun:
            tengxun_time[xh] += 1
            tengxun_period[xh] += period

app_data['zhibo_time'] = pd.Series(list(zhibo_time.values()))
app_data['duanshipin_time'] = pd.Series(list(duanshipin_time.values()))
app_data['shipin_time'] = pd.Series(list(shipin_time.values()))
app_data['youxi_time'] = pd.Series(list(youxi_time.values()))
app_data['tengxun_time'] = pd.Series(list(tengxun_time.values()))

app_data['zhibo_period'] = pd.Series(list(zhibo_period.values()))
app_data['duanshipin_period'] = pd.Series(list(duanshipin_period.values()))
app_data['shipin_period'] = pd.Series(list(shipin_period.values()))
app_data['youxi_period'] = pd.Series(list(youxi_period.values()))
app_data['tengxun_period'] = pd.Series(list(tengxun_period.values()))

print(app_data.head())

app_data.to_csv('../Data/app_use.csv',index=None,encoding='utf-8')