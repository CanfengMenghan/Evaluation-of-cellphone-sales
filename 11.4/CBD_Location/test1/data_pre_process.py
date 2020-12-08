import pandas as pd
import re
import csv
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import train_test_split

# data1 = pd.read_csv('D:\Alibaba game\CBD_location\ccf_first_round_user_shop_behavior.csv', 'r')
# data2 = pd.read_csv('D:\Alibaba game\CBD_location\ccf_first_round_shop_info.csv', 'r')
# data1.info()
# data2.info()

# data1.drop(['user_id'], axis=1, inplace=True)


# with open('D:\Alibaba game\CBD_location\c_test.csv', 'r+') as data:
#     data_csv = csv.reader(data)
#     data_iter = list(data_csv)
#     df = pd.DataFrame(data_iter)
#     df['wifi_infos'].str.split('/', expand=True).stack()
#     print(df)

# data = pd.read_csv('D:\Alibaba game\CBD_location\c_test.csv')
# data=csv.reader(open('D:\Alibaba game\CBD_location\c_test.csv','r'))

# with open('D:\Alibaba game\CBD_location\ccf_first_round_user_shop_behavior.csv', 'r') as csvfile:
#     for i in csvfile.readlines():
#         info = i.split(',')[5]
#         info1 = re.split('[|;]', str(info))
#         out = open('D:\Alibaba game\CBD_location\_processing.csv', 'a', newline='')
#         write = csv.writer(out)
#         write.writerow(info1)
#         print(info1)
#         print("write over")


# train = pd.read_csv('D:\Alibaba game\CBD_location\c_test.csv', 'r')
# train_xy, val = train_test_split(train, test_size=0.3, random_state=1)
# print(val)
# print("!!!!!!!!!")
# print(train_xy)

# for info in data:
#     info1 = re.split('[;|]', info)
# print(data)
# for row in data:
#     info = re.split('[;|]', str(row))
#     print(info)
#     out = open('D:\Alibaba game\CBD_location\_processing.csv','a',newline='')
#     write = csv.writer(out)
#     write.writerow(info)
#     print("write over")


# a = pd.read_csv('D:\Alibaba game\CBD_location\c_test.csv', 'r')
# test = pd.DataFrame(a)
# print("!!!!")
# test.rename(columns={'user_id':'123'},inplace=True)
#
# print(test)


with open(r'D:\Alibaba game\CBD_location\AB_evaluation_public.csv') as csvfile:
    for i in csvfile.readlines():
        info = i.split(',')[6]
        info1 = re.split('[|;]', str(info))
        out = open(r'D:\Alibaba game\CBD_location\test_AB_pre.csv', 'a', newline='')
        write = csv.writer(out)
        write.writerow(info1)
        print("write over")
