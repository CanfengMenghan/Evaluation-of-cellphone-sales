import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV
import time


start_time = time.time()
train = pd.read_csv(r"D:\Alibaba game\CBD_location\train.csv")
tests = pd.read_csv(r"D:\Alibaba game\CBD_location\test.csv")

print(train.info())
print("///////////////")
train['time_stamp'] = pd.to_datetime(pd.Series(train['time_stamp']))
tests['time_stamp'] = pd.to_datetime(pd.Series(tests['time_stamp']))
# print(type(train.ix[0,"time_stamp"]))

train['Year'] = train['time_stamp'].apply(lambda x: x.year)
train['Month'] = train['time_stamp'].apply(lambda x: x.year)
train['weekday'] = train['time_stamp'].apply(lambda x: x.year)
tests['Year'] = tests['time_stamp'].apply(lambda x: x.year)
tests['Month'] = tests['time_stamp'].apply(lambda x: x.year)
tests['weekday'] = tests['time_stamp'].apply(lambda x: x.year)

train = train.drop('time_stamp', axis=1)
train = train.dropna(axis=0)
tests = tests.drop('time_stamp', axis=1)
tests = tests.drop('row_id', axis=1)
tests = tests.fillna(method='pad')
train = train.fillna(method='pad')
# a = set(train.columns)
# b = set(tests.columns)
# print(b)

# print(train)
# print("000000000000000000000000000华丽丽的分割线0000000000000000000000000000")
# print(tests)
# train = train.fillna(-999)
# test = tests.fillna(-999)

features = list(train.columns[1:])
# print(features)
tests.ix[0, "shop_id"]=str(tests.ix[0, "shop_id"])
# print(type(tests.ix[0, "shop_id"]))
for f in train.columns:
    if train[f].dtype == 'object':
        print(f)
        if f != 'shop_id':
            label = preprocessing.LabelEncoder()
            label.fit(list(train[f].values)+list(tests[f].values))
            train[f] = label.transform(list(train[f].values))
            tests[f] = label.transform(list(tests[f].values))
        label = preprocessing.LabelEncoder()
        label.fit(list(train[f].values)+list(tests[f].values))
        train[f] = label.transform(list(train[f].values))
        tests[f] = label.transform(list(tests[f].values))

    elif train[f].dtype == 'bool':
        # print(f)
        label = preprocessing.LabelEncoder()
        label.fit(list(train[f].values) + list(tests[f].values))
        train[f] = label.transform(list(train[f].values))
        tests[f] = label.transform(list(tests[f].values))
# print(train)
# print(tests)
# print(type(train.ix[0,"con_sta1"]))
print(train[:10])
X = train.drop(['shop_id'],axis = 1)
y = train.shop_id.values


train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.05, random_state=144)
d_train = lgb.Dataset(train_X, label=train_y, max_bin=8192)
d_valid = lgb.Dataset(valid_X, label=valid_y, max_bin=8192)
watchlist = [d_train, d_valid]

params = {
    'learning_rate': 0.75,
    'application': 'multiclass',
    'max_depth': 3,
    'num_leaves': 100,
    'verbosity': -1,
    'metric': 'multi_error',
    'nthread': 4,
}


model = lgb.train(params, train_set=d_train, num_boost_round=1000, valid_sets=watchlist, \
                  early_stopping_rounds=100, verbose_eval=10)
preds = model.predict(tests)

print('[{}] Predict lgb completed.'.format(time.time() - start_time))

submission = pd.DataFrame(preds)
submission.to_csv("submission_lgbm.csv", index=False)