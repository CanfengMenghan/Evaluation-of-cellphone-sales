import pandas as pd

shop_info = pd.read_csv('D:\Alibaba game\CBD_location\ccf_first_round_shop_info.csv')
user_shop_info = pd.read_csv('D:\Alibaba game\CBD_location\ccf_first_round_shop_info.csv')
data=pd.merge(user_shop_info,shop_info,how='left',on='shop_id')
test_data = pd.read_csv("D:\Alibaba game\CBD_location\AB_evaluation_public")


def extractData(x):
    wifi = {}
    wifi_infos = x.split(';')
    for info in wifi_infos:
        info = info.split('|')
        wifi[info[0]] = -int(info[1])
    return wifi


data['wifi_infos_extract'] = data['wifi_infos'].map(extractData)
target = user_shop_info['shop_id']

print("交易记录：%d\n店铺数：%d\n测试数据数：%d" % (len(user_shop_info), len(shop_info), len(test_data)))