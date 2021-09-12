import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.corpus import stopwords
import jieba as jb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

train = pd.read_csv('train.csv',encoding='utf-8')
train=train.dropna()

print(len(train))
print(train.sample(10))

#标签
label=[]
for i in train['accusation']:
    if ';' in i:
        i=i.split(';')[0]
        label.append(i)
    else:
        label.append(i)
train['accusation']=label
print(train['accusation'])

#标签字典
tags = train['accusation'].values
tag_dic={}

for tag in tags:
    if tag not in tag_dic:
        tag_dic[tag]=1
    else:
         tag_dic[tag]+=1
df = pd.DataFrame(list(tag_dic.items()), columns=['tag', 'count']).sort_values(by = 'count',axis = 0,ascending = False)
print('标签总数:',len(df))
print(df.head(10))

# 定义删除除字母,数字，汉字以外的所有符号的函数
def remove_punctuation(line):
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)
    return line

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

stopwords = stopwordslist("stopwords.txt")

#定义数据清洗函数
def text_prepare(text):
    text = remove_punctuation(text)
    text=" ".join([w for w in list(jb.cut(text)) if w not in stopwords])
    return text

#划分训练集、验证集
X_train, X_val, y_train, y_val = train_test_split(train['fact'].values, train['accusation'].values, test_size=0.3)

# 开始进行数据清洗
X_train = [text_prepare(x) for x in X_train]
X_val = [text_prepare(x) for x in X_val]

print(X_train[:2])
print(y_train[:2])

print('KEY:',tag_dic)

#f1值
def print_evaluation_scores(y_val, predicted):
    accuracy = accuracy_score(y_val, predicted)
    f1_score_macro = f1_score(y_val, predicted, average='macro')
    f1_score_micro = f1_score(y_val, predicted, average='micro')
    f1_score_weighted = f1_score(y_val, predicted, average='weighted')
    print("accuracy:", accuracy)
    print("f1_score_macro:", f1_score_macro)
    print("f1_score_micro:", f1_score_micro)
    print("f1_score_weighted:", f1_score_weighted)

# CV+多项式朴素贝叶斯的多分类模型
NB_pipeline = Pipeline([
    ('cv', CountVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2))),#过滤掉词频小于5次的单词(绝对值)，过滤掉大于那些在90%以上文档中都出现的单词(相对值)，选用1-2个词进行前后的组合，构成新的标签值
    ('clf', OneVsRestClassifier(MultinomialNB(
        fit_prior=True, class_prior=None))),#考虑先验概率，不考虑用户自己输入的先验概率
])

NB_pipeline.fit(X_train, y_train)
predicted = NB_pipeline.predict(X_val)
print_evaluation_scores(y_val, predicted)

print(predicted)
print(len(predicted))

test = pd.read_csv('test.csv',encoding='utf-8')
test['fact'] = [text_prepare(x) for x in test['fact']]
print(test['fact'][:2])
test_predicted = np.array(NB_pipeline.predict(test['fact']))

print(len(test_predicted))
print(test_predicted)


result=pd.read_csv('test.csv',encoding='utf-8')
result['accusation']=test_predicted

order=['ids','accusation']
result=result[order]
result.to_csv('result.csv', encoding='utf-8',index=False)
