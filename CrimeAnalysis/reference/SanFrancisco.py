import pandas as pd
import numpy as np

# 1、载入数据
train = pd.read_csv('dataset/train.csv', parse_dates = ['Dates'])
test = pd.read_csv('dataset/test.csv', parse_dates = ['Dates'])

# 2、数据预处理，对category进行编码
from sklearn import preprocessing
label = preprocessing.LabelEncoder()
crime = label.fit_transform(train.Category) #进行编号

# 3、对Dates、DayOfWeek、PdDistrict三个特征进行二值化处理，因为3个在训练集和测试集都出现
days = pd.get_dummies(train.DayOfWeek)
district = pd.get_dummies(train.PdDistrict)
hour = pd.get_dummies(train.Dates.dt.hour)

train_data = pd.concat([days, district, hour], axis=1)   # 将days district hour连成一张表 ，当axis = 1的时候，concat就是行对齐，然后将不同列名称的两张表合并
train_data['crime'] = crime  # 在DataFrame数据结构 表的 最后加一列，在本例中相当于标签
# 实际上，只使用了三个特征，和犯罪类型作为标签 即只使用了原始数据集中的4列数据
# 但是train_data这张表  其实是将3个特征展开成了几十个特征 对应一个标签


# 针对测试集做同样的处理
days = pd.get_dummies(test.DayOfWeek)
district = pd.get_dummies(test.PdDistrict)
hour = pd.get_dummies(test.Dates.dt.hour)
test_data = pd.concat([days, district, hour], axis=1)

# 4、将样本几何分割成训练集和验证集(70%训练,30%验证)，返回的是划分好的训练集 和 验证集
from sklearn.cross_validation import train_test_split
training, validation = train_test_split(train_data, train_size=0.7)


# 5、朴素贝叶斯
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB()
feature_list = training.columns.tolist()   #将列名字转换为列表
feature_list = feature_list[:len(feature_list) - 1]   # 选取的特征列  最后一列是标签，不能要，注意列表是左闭右开
model.fit(training[feature_list], training['crime'])    #根据给定的训练数据拟合模型

predicted = np.array(model.predict_proba(validation[feature_list]))   #validation[feature_list] 不包括最后一列crime 的验证集    model.predict_proba 第 i 行 第 j 列上的数值是模型预测第 i 个预测样本 为某个【标签】的概（表头是标签类别），从小到大排序的    predicted是在验证集上的结果
print ("朴素贝叶斯log损失为 %f" % (log_loss(validation['crime'], predicted)))   #多分类的对数损失

# 6、其他模型等 （逻辑回归，随机森林）
from sklearn.linear_model import LogisticRegression
model_LR = LogisticRegression(C=0.1)
model_LR.fit(training[feature_list], training['crime'])
predicted = np.array(model_LR.predict_proba(validation[feature_list]))
print ("逻辑回归log损失为 %f" %(log_loss(validation['crime'], predicted)))

from sklearn.ensemble import RandomForestClassifier
model_RF = RandomForestClassifier()
model_RF.fit(training[feature_list], training['crime'])
predicted = np.array(model_RF.predict_proba(validation[feature_list]))
print ("随机森林log损失为 %f" %(log_loss(validation['crime'], predicted)))

# 7、在测试集上运行
test_predicted = np.array(model.predict_proba(test_data[feature_list])) # model为朴素贝叶斯

# 8、保存结果
col_names = np.sort(train['Category'].unique())  # 唯一，按首字母从小到大排序
result = pd.DataFrame(data=test_predicted, columns=col_names)  # 合成DataFrame数据结构的表 col_names是排序的，test_predicted由于predict_proba，所以也是按顺序的
result['Id'] = test['Id'].astype(int) # 从 dtype: int64 变为 dtype: int32 并且在最后加一列result['Id']
result.to_csv('test_output.csv', index=False)  #保存
print ("finish")

