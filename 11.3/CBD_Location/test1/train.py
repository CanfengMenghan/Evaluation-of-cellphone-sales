import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

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
        # print(f)
        # if f != 'shop_id':
        #     label = preprocessing.LabelEncoder()
        #     label.fit(list(train[f].values)+list(tests[f].values))
        #     train[f] = label.transform(list(train[f].values))
        #     tests[f] = label.transform(list(tests[f].values))
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

parameters = {
              'objective': ['multi:softmax'],
              'learning_rate': [0.05], # so called `eta` value
              'max_depth': [6],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [5], # number of trees, change it to 1000 for better results
              'missing': [-999],
              'seed': [1000]
              }

xgb_model = xgb.XGBClassifier()
if __name__ == '__main__':
    clf = GridSearchCV(xgb_model, param_grid=parameters, n_jobs=5,
                       cv=StratifiedKFold(train['shop_id'], n_folds=2,
                                          shuffle=True),
                       # scoring='roc_auc',
                       verbose=2, refit=True)
# if __name__ == '__main__':
    clf.fit(train[features], train["shop_id"])

    best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
    # print('RAW AUC score:', score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
    test_probs = clf.predict_proba(tests[features])[:, 1]

    sample = pd.read_csv(r"D:\Alibaba game\CBD_location\sample_submission.csv")
    sample.shop_id = test_probs
    sample.to_csv("xgboost_best_parameter_submission.csv", index=False)


time = time.time()-start_time
print("running time:", time, "seconds")
print("/////////////////////////////////////////////////////////////////////")
#
#
# plst = list(parameters.items())
#
# offset = 35000
#
# num_rounds = 100
# xgtest = xgb.DMatrix(tests)
#
#
# # 划分训练集与验证集
# xgtrain = xgb.DMatrix(train[:offset,:])
# xgval = xgb.DMatrix(train[offset:, :])
#
# # return 训练和验证的错误率
# watchlist = [(xgtrain, 'train'), (xgval, 'val')]
#
#
# # training model
# # early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
# model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
# # model.save_model('./model/xgb.model') # 用于存储训练出的模型
# preds = model.predict(xgtest, ntree_limit = model.best_iteration)
#
# # 将预测结果写入文件，方式有很多，自己顺手能实现即可
# np.savetxt('submission_xgb_MultiSoftmax.csv',np.c_[range(1,len(test)+1),preds],
#                 delimiter=',',header='ImageId,Label',comments='',fmt='%d')
#
#
# cost_time = time.time()-start_time
# print("end ......", '\n',"cost time:",cost_time,"(s)......")
#
