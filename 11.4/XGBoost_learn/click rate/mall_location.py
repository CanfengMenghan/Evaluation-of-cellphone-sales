import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
import numpy as np

train = pd.read_csv(r'D:\XGBoost_learn\click rate\train1.csv', header=0)
tests = pd.read_csv(r'D:\XGBoost_learn\click rate\test_pre.csv', header=0)
# trains=train.iloc[:, 1:].values
# labels=train.iloc[:,:1].values
# test = tests.iloc[:, :].values
'''
train['time_stamp'] = pd.to_datetime(pd.Series(train['time_stamp']))
tests['time_stamp'] = pd.to_datetime(pd.Series(tests['time_stamp']))
train['Year'] = train['time_stamp'].apply(lambda x: x.year)#Year
train['Month'] = train['time_stamp'].apply(lambda x: x.month)#Month
train['weekday'] = train['time_stamp'].dt.dayofweek#weekday
train['time'] = train['time_stamp'].dt.time#time
tests['Year'] = tests['time_stamp'].apply(lambda x: x.year)#Year
tests['Month'] = tests['time_stamp'].apply(lambda x: x.month)#Month
tests['weekday'] = tests['time_stamp'].dt.dayofweek#weekday
tests['time'] = tests['time_stamp'].dt.time#time
train = train.drop('time_stamp', axis=1)
train = train.dropna(axis=0)
tests = tests.drop('time_stamp', axis=1)
tests = tests.fillna(method='pad')
'''

for f in train.columns:
    if train[f].dtype=='object':
        if f != 'shop_id':
            print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[f].values))
            train[f] = lbl.transform(list(train[f].values))

for f in tests.columns:
    if tests[f].dtype == 'object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(tests[f].values))
        tests[f] = lbl.transform(list(tests[f].values))
print("test")
print(tests.info())
# for f in train.columns:
#     if f !='':
#         train[f] = train[f].astype(float)

print(train.info())
# train = train.astype(float)
# tests = tests.astype(float)
trains = train.iloc[:, 1:].values
labels = train.iloc[:, :1].values
test = tests.iloc[:, 1:].values

feature_columns_to_use = ['wifi_strong1','wifi_strong2','wifi_strong3']

big_X = train[feature_columns_to_use].append(tests[feature_columns_to_use])
train_X = big_X[0:train.shape[0]].as_matrix()
test_X = big_X[train.shape[0]::].as_matrix()
train_y = train['shop_id']
gbm = xgb.XGBClassifier(silent=1, max_depth=3, n_estimators=300, learning_rate=0.05)
gbm.fit(train_X, train_y)
predictions = gbm.predict(test_X)
submission = pd.DataFrame({'row_id': tests['row_id'],
                            'shop_id': predictions})
print(submission)
submission.to_csv("submission.csv", index=False)
'''
print(trains)
parameters={
    'silent':1,
    'max_depth': 3,
    'n_estimators':300,
    'learning_rate':0.005,
}
feature_types={
    'float','str','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float','float',
}
# feature_types = {
#     'str', 'float',
# }
ft=list(feature_types)
ParamLst = dict(parameters.items())

offset = 15

num_rounds = 1
xgtest = xgb.DMatrix(tests)

print("//////////////////////////")
# 划分训练集与验证集
xgtrain = xgb.DMatrix(trains[:offset,:], label=labels[:offset])
print("//////////////////////////")
xgval = xgb.DMatrix(trains[offset:, :], label=labels[offset:])

# return 训练和验证的错误率
watchlist = [(xgtrain, 'train'), (xgval, 'val')]


# training model
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(ParamLst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
# model.save_model('./model/xgb.model') # 用于存储训练出的模型
preds = model.predict(xgtest, ntree_limit=model.best_iteration)

# 将预测结果写入文件，方式有很多，自己顺手能实现即可
np.savetxt('submission_xgb_MultiSoftmax.csv', np.c_[range(1, len(test)+1), preds],
                delimiter=',', header='ImageId,Label', comments='', fmt='%d')
'''
