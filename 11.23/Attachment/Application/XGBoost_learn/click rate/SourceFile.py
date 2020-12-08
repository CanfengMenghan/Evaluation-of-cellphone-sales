import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.base import TransformerMixin
from sklearn import preprocessing
train = pd.read_csv(r'D:\XGBoost_learn\mall_location\train2.csv', header=0)
tests = pd.read_csv(r'D:\XGBoost_learn\mall_location\test_pre.csv', header=0)

# We'll impute missing values using the median for numeric columns and the most
# common value for string columns.
# This is based on some nice code by 'sveitser' at http://stackoverflow.com/a/25562948

# #######################    时间数据预处理    #########################
# print(train.info())
# print(tests.info())
train['time_stamp'] = pd.to_datetime(pd.Series(train['time_stamp']))
tests['time_stamp'] = pd.to_datetime(pd.Series(tests['time_stamp']))
# print(type(train.ix[0,"time_stamp"]))

train['Year'] = train['time_stamp'].apply(lambda x: x.year)
train['Month'] = train['time_stamp'].apply(lambda x: x.month)
train['weekday'] = train['time_stamp'].dt.dayofweek
train['time'] = train['time_stamp'].dt.time
tests['Year'] = tests['time_stamp'].apply(lambda x: x.year)
tests['Month'] = tests['time_stamp'].apply(lambda x: x.month)
tests['weekday'] = tests['time_stamp'].dt.dayofweek
tests['time'] = tests['time_stamp'].dt.time

train = train.drop('time_stamp', axis=1)
train = train.dropna(axis=0)
tests = tests.drop('time_stamp', axis=1)
tests = tests.fillna(method='pad')
# print(train)
train.ix[0, "shop_id"]=str(train.ix[0, "shop_id"])

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
print(train.info())


class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        # 这一句话就非常拗口，意思是如果X[c]的类型是object（‘O’表示object）的话
        # 就将[X[c].value_counts().index[0]传给空值，[X[c].value_counts().index[0]表示的是重复出现最多的那个数，
        # 如果不是object类型的话，就传回去X[c].median()，也就是这些数的中位数
        for c in X:
            if X[c].dtype == np.dtype('O'):
                fill_number = X[c].value_counts().index[0]
                self.fill = pd.Series(fill_number, index=X.columns)
            else:
                fill_number = X[c].median()
            self.fill = pd.Series(fill_number, index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


feature_columns_to_use = ['Year', 'Month', 'weekday', 'time', 'longitude', 'latitude',
                          'wifi_id1', 'wifi_strong1', 'con_sta1',
                          'wifi_id2', 'wifi_strong2', 'con_sta2',
                          'wifi_id3', 'wifi_strong3', 'con_sta3',
                          'wifi_id4', 'wifi_strong4', 'con_sta4',
                          'wifi_id5', 'wifi_strong5', 'con_sta5',
                          'wifi_id6', 'wifi_strong6', 'con_sta6',
                          'wifi_id7', 'wifi_strong7', 'con_sta7',
                          'wifi_id8', 'wifi_strong8', 'con_sta8',
                          'wifi_id9', 'wifi_strong9', 'con_sta9',
                          'wifi_id10', 'wifi_strong10', 'con_sta10',
                          ]
nonnumeric_columns = [    'wifi_id1',
                          'wifi_id2',
                          'wifi_id3',
                          'wifi_id4',
                          'wifi_id5',
                          'wifi_id6',
                          'wifi_id7',
                          'wifi_id8',
                          'wifi_id9',
                          'wifi_id10']

# Join the features from train and test together before imputing missing values,
# in case their distribution is slightly different
big_X = train[feature_columns_to_use].append(tests[feature_columns_to_use])
big_X_imputed = DataFrameImputer().fit_transform(big_X)
print("////////////////////////////////this is big_x_imputed://////////////////////////////////////")
# print(big_X_imputed)
print("////////////////////////////////////////////////")
# XGBoost doesn't (yet) handle categorical features automatically, so we need to change
# them to columns of integer values.
# See http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing for more
# details and options
# le = LabelEncoder()
for feature in nonnumeric_columns:
    big_X_imputed[feature] = LabelEncoder().fit_transform(big_X_imputed[feature])
# print("///////////////////////////////////////////////////")
# Prepare the inputs for the model
train_X = big_X_imputed[0:train.shape[0]].as_matrix()
test_X = big_X_imputed[train.shape[0]::].as_matrix()
train_y = train['shop_id']

gbm = xgb.XGBClassifier(silent=1, max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)
print(r">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(gbm)
predictions = gbm.predict(test_X)

submission = pd.DataFrame({'row_id': tests['row_id'],
                            'shop_id': predictions})

submission.to_csv("submission.csv", index=False)
