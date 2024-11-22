# AB测试是为检验某项改动（上线功能）是否能对指标带来提升，而该项目中指标的定义还需根据具体的业务进行补充（同环比、GMV、点击率转换率等）
# AB测试解决是的决策的准确性和成本之间的平衡问题，一旦开始使用它，预示着团队以数据目标为导向的决心

#确定目标和假设->确定指标->（确定实验单位）->计算样本量->实施测试->分析实验结果

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygwalker as pyg
from statsmodels.stats.power import NormalIndPower


data_order = pd.read_csv('D:\\resume\\AB test\\order.csv')
data_menu = pd.read_csv('D:\\resume\\AB test\\menu.csv')
data_detail = pd.read_csv('D:\\resume\\AB test\\detail.csv')
pd.set_option('display.max_columns', None)#展示全部列

data = pd.merge(pd.merge(data_order,data_detail,how='right',on='order_id'),data_menu,how='left',left_on='item_id',right_on='product_id')

# 去重——其实不用去重，每个订单可以点重复的菜
# data.duplicated()
# data = data.drop_duplicates(inplace = False)#这样就不会在原地修改 data，而是返回一个新的、去除了重复行的 DataFrame
# data

#print(data.isnull().any())#查看缺失值
data = data.drop(columns=['product_id', 'price_y'])
#data = data.drop('product_id',axis=1).drop('price_y',axis=1)#另一种去除指定列的方法

data.rename(columns={'price_x':'price'},inplace= True )
data = data[['date','order_id','item_id','price','product_name','category','total']]

#修改字段
# 1.自建词典
# 2.词典映射
productTranslation_dict = {
    'Steamed Red Bean Rick Cake': '红豆糯米糕',
    'Chicken Xiaolongbao': '鸡肉小笼包',
    'Steamed Red Bean Rice Cake with Walnuts': '核桃红豆年糕',
    'Shredded Pork Fried Rice': '肉丝炒饭',
    'Steamed Shrimp and Pork Dumplings': '虾肉猪肉水饺',
    'Shrimp and Shredded Pork Fried Rice': '虾肉肉丝炒饭',
    'Steamed Fish Dumplings': '鱼肉水饺',
    'Shrimp Fried Rice': '虾肉炒饭',
    'Steamed Chinese Style Layer Cake': '中国风蛋糕',
    'Green Squash and Shrimp Xiaolongbao': '青南瓜虾肉小笼包',
    'Pork Xiaolongbao': '猪肉小笼包',
    'Steamed Vegetable and Ground Pork Dumplings': '蔬菜猪肉水饺',
    'Vegetarian Mushroom Buns': '素馅馒头',
    'Vegetable and Ground Pork Buns': '蔬菜猪肉馅包',
    'Pork Buns': '猪肉包子',
    'Crab Roe and Pork Xiaolongbao': '蟹黄猪肉小笼包',
    'Steamed Vegetarian Mushroom Dumplings': '素馅蒸饺'
}
categoryTranslation_dict = {
    'Desserts': '甜点',
    'Xiaolongbao': '小笼包',
    'Fried Rice': '炒饭',
    'Dumplings & Shao Mai': '饺子与烧卖',
    'Buns': '馒头'
}

data['product_name']=data['product_name'].map(productTranslation_dict)
data['category']=data['category'].map(categoryTranslation_dict)
# print(data.head())

#指标建立（格局业务场景简历指标）tip：常规——总量、平均、环比、分类（不同...的...）
# 订单数量order
# 点菜数量item
# 每个订单平均点菜数量avg_item、菜价avg_itemPrice
# 销售额sales、订单平均销售额avg_sales
# 不同菜系在当日销售的数量和销售额占比
# 不同菜品在当日销售的数量和销售额占比

df = pd.DataFrame(data.groupby('date').agg(order = ('order_id', pd.Series.nunique)))#聚合
df = df.rename_axis('date').reset_index() #默认为整数索引
# print(df.head())
df['item'] = data.groupby('date')['item_id'].count().values
# print(df.head())
df['avg_item'] = df.apply(lambda x: round(x['item']/x['order'],2), axis=1)#apply()函数和lambda()函数运用
df['sales'] = data.groupby('date')['price'].sum().values
df['avg_itemPrice'] = df.apply(lambda x: round(x['sales']/x['item'],2), axis=1)
df['avg_sales']=df.apply(lambda x: round(x['sales']/x['order'],2), axis=1)
# print(df.head())

category_array = data['category'].unique()
product_array = data['product_name'].unique()

for c in category_array:
    str1 = c + '数量占比'
    str2 = c + '销售额占比'
    temp1 = data.groupby('date')['category'].apply(lambda x: (x == c).sum()).reset_index(name='count')
    temp2 = data.groupby('date').apply(lambda x: x[x.category == c]['price'].sum()).reset_index(drop=True)

    df[str1] = round(temp1['count'] / df['item'], 4)
    df[str2] = round(temp2 / df['sales'], 4)

# 需要删除两个异常值
df.drop(index = df[df['date']=='2023-03-31'].index[0],inplace=True)
df.drop(index = df[df['date']=='2023-05-17'].index[0],inplace=True)


# vis_spec = r"""{"config":[{"config":{"defaultAggregated":true,"geoms":["line"],"coordSystem":"generic","limit":-1},"encodings":{"dimensions":[{"dragId":"gw_udlW","fid":"date","name":"date","basename":"date","semanticType":"nominal","analyticType":"dimension"},{"dragId":"gw_mVox","fid":"今日最爱菜品","name":"今日最爱菜品","basename":"今日最爱菜品","semanticType":"nominal","analyticType":"dimension"},{"dragId":"gw_hMZd","fid":"今日最爱品类","name":"今日最爱品类","basename":"今日最爱品类","semanticType":"nominal","analyticType":"dimension"},{"dragId":"gw_mea_key_fid","fid":"gw_mea_key_fid","name":"Measure names","analyticType":"dimension","semanticType":"nominal"}],"measures":[{"dragId":"gw_1xQz","fid":"甜点数量占比","name":"甜点数量占比","basename":"甜点数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_Wyge","fid":"甜点销售额占比","name":"甜点销售额占比","basename":"甜点销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_d0Ni","fid":"order","name":"order","basename":"order","semanticType":"quantitative","analyticType":"measure"},{"dragId":"gw_KHzG","fid":"item","name":"item","basename":"item","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_HV6a","fid":"sales","name":"sales","basename":"sales","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_PurS","fid":"avg_item","name":"avg_item","basename":"avg_item","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_b3ba","fid":"avg_itemPrice","name":"avg_itemPrice","basename":"avg_itemPrice","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_spkK","fid":"avg_sales","name":"avg_sales","basename":"avg_sales","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_bSj1","fid":"小笼包数量占比","name":"小笼包数量占比","basename":"小笼包数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_bItY","fid":"小笼包销售额占比","name":"小笼包销售额占比","basename":"小笼包销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_JQpv","fid":"炒饭数量占比","name":"炒饭数量占比","basename":"炒饭数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_aU3D","fid":"炒饭销售额占比","name":"炒饭销售额占比","basename":"炒饭销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_XRSg","fid":"饺子与烧卖数量占比","name":"饺子与烧卖数量占比","basename":"饺子与烧卖数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_kXqb","fid":"饺子与烧卖销售额占比","name":"饺子与烧卖销售额占比","basename":"饺子与烧卖销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_mnby","fid":"馒头数量占比","name":"馒头数量占比","basename":"馒头数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_krws","fid":"馒头销售额占比","name":"馒头销售额占比","basename":"馒头销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_bWRo","fid":"红豆糯米糕数量占比","name":"红豆糯米糕数量占比","basename":"红豆糯米糕数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_Hi-a","fid":"红豆糯米糕销售额占比","name":"红豆糯米糕销售额占比","basename":"红豆糯米糕销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_zKOB","fid":"鸡肉小笼包数量占比","name":"鸡肉小笼包数量占比","basename":"鸡肉小笼包数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_GtpD","fid":"鸡肉小笼包销售额占比","name":"鸡肉小笼包销售额占比","basename":"鸡肉小笼包销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_wqPH","fid":"核桃红豆年糕数量占比","name":"核桃红豆年糕数量占比","basename":"核桃红豆年糕数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_I7il","fid":"核桃红豆年糕销售额占比","name":"核桃红豆年糕销售额占比","basename":"核桃红豆年糕销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_2RiB","fid":"肉丝炒饭数量占比","name":"肉丝炒饭数量占比","basename":"肉丝炒饭数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_RNWr","fid":"肉丝炒饭销售额占比","name":"肉丝炒饭销售额占比","basename":"肉丝炒饭销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_PRvh","fid":"虾肉猪肉水饺数量占比","name":"虾肉猪肉水饺数量占比","basename":"虾肉猪肉水饺数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_hZ4H","fid":"虾肉猪肉水饺销售额占比","name":"虾肉猪肉水饺销售额占比","basename":"虾肉猪肉水饺销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_uMlO","fid":"虾肉肉丝炒饭数量占比","name":"虾肉肉丝炒饭数量占比","basename":"虾肉肉丝炒饭数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_pghw","fid":"虾肉肉丝炒饭销售额占比","name":"虾肉肉丝炒饭销售额占比","basename":"虾肉肉丝炒饭销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_Tq4h","fid":"鱼肉水饺数量占比","name":"鱼肉水饺数量占比","basename":"鱼肉水饺数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_zuE_","fid":"鱼肉水饺销售额占比","name":"鱼肉水饺销售额占比","basename":"鱼肉水饺销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_syEO","fid":"虾肉炒饭数量占比","name":"虾肉炒饭数量占比","basename":"虾肉炒饭数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_IBQC","fid":"虾肉炒饭销售额占比","name":"虾肉炒饭销售额占比","basename":"虾肉炒饭销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_gpAE","fid":"中国风蛋糕数量占比","name":"中国风蛋糕数量占比","basename":"中国风蛋糕数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_lBaB","fid":"中国风蛋糕销售额占比","name":"中国风蛋糕销售额占比","basename":"中国风蛋糕销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_8qOM","fid":"青南瓜虾肉小笼包数量占比","name":"青南瓜虾肉小笼包数量占比","basename":"青南瓜虾肉小笼包数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_FoYA","fid":"青南瓜虾肉小笼包销售额占比","name":"青南瓜虾肉小笼包销售额占比","basename":"青南瓜虾肉小笼包销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_D68G","fid":"猪肉小笼包数量占比","name":"猪肉小笼包数量占比","basename":"猪肉小笼包数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_tDWx","fid":"猪肉小笼包销售额占比","name":"猪肉小笼包销售额占比","basename":"猪肉小笼包销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_mhYW","fid":"蔬菜猪肉水饺数量占比","name":"蔬菜猪肉水饺数量占比","basename":"蔬菜猪肉水饺数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_sqjH","fid":"蔬菜猪肉水饺销售额占比","name":"蔬菜猪肉水饺销售额占比","basename":"蔬菜猪肉水饺销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_dlyj","fid":"素馅馒头数量占比","name":"素馅馒头数量占比","basename":"素馅馒头数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_uzjP","fid":"素馅馒头销售额占比","name":"素馅馒头销售额占比","basename":"素馅馒头销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_CEez","fid":"蔬菜猪肉馅包数量占比","name":"蔬菜猪肉馅包数量占比","basename":"蔬菜猪肉馅包数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_TaP6","fid":"蔬菜猪肉馅包销售额占比","name":"蔬菜猪肉馅包销售额占比","basename":"蔬菜猪肉馅包销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_qS7D","fid":"猪肉包子数量占比","name":"猪肉包子数量占比","basename":"猪肉包子数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_gIw7","fid":"猪肉包子销售额占比","name":"猪肉包子销售额占比","basename":"猪肉包子销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_WdlX","fid":"蟹黄猪肉小笼包数量占比","name":"蟹黄猪肉小笼包数量占比","basename":"蟹黄猪肉小笼包数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_Cpgp","fid":"蟹黄猪肉小笼包销售额占比","name":"蟹黄猪肉小笼包销售额占比","basename":"蟹黄猪肉小笼包销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_48C5","fid":"素馅蒸饺数量占比","name":"素馅蒸饺数量占比","basename":"素馅蒸饺数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_u3g3","fid":"素馅蒸饺销售额占比","name":"素馅蒸饺销售额占比","basename":"素馅蒸饺销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_count_fid","fid":"gw_count_fid","name":"Row count","analyticType":"measure","semanticType":"quantitative","aggName":"sum","computed":true,"expression":{"op":"one","params":[],"as":"gw_count_fid"}},{"dragId":"gw_mea_val_fid","fid":"gw_mea_val_fid","name":"Measure values","analyticType":"measure","semanticType":"quantitative","aggName":"sum"}],"rows":[{"dragId":"gw_5SC4","fid":"order","name":"order","basename":"order","semanticType":"quantitative","analyticType":"measure"},{"dragId":"gw_LbrM","fid":"item","name":"item","basename":"item","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_sqog","fid":"avg_item","name":"avg_item","basename":"avg_item","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_5mlh","fid":"avg_itemPrice","name":"avg_itemPrice","basename":"avg_itemPrice","analyticType":"measure","semanticType":"quantitative","aggName":"sum"}],"columns":[{"dragId":"gw_k6ci","fid":"date","name":"date","basename":"date","semanticType":"nominal","analyticType":"dimension"}],"color":[],"opacity":[],"size":[],"shape":[],"radius":[],"theta":[],"longitude":[],"latitude":[],"geoId":[],"details":[],"filters":[],"text":[]},"layout":{"showActions":false,"showTableSummary":false,"stack":"stack","interactiveScale":false,"zeroScale":true,"size":{"mode":"auto","width":320,"height":200},"format":{},"geoKey":"name","resolve":{"x":false,"y":false,"color":false,"opacity":false,"shape":false,"size":false}},"visId":"gw_V-aw","name":"Chart 1"},{"config":{"defaultAggregated":true,"geoms":["line"],"coordSystem":"generic","limit":-1},"encodings":{"dimensions":[{"dragId":"gw_5gyy","fid":"date","name":"date","basename":"date","semanticType":"nominal","analyticType":"dimension"},{"dragId":"gw_kHJ5","fid":"order","name":"order","basename":"order","semanticType":"quantitative","analyticType":"dimension"},{"dragId":"gw_bYOk","fid":"avg_item","name":"avg_item","basename":"avg_item","analyticType":"dimension","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_PTqU","fid":"今日最爱菜品","name":"今日最爱菜品","basename":"今日最爱菜品","semanticType":"nominal","analyticType":"dimension"},{"dragId":"gw_qsZn","fid":"今日最爱品类","name":"今日最爱品类","basename":"今日最爱品类","semanticType":"nominal","analyticType":"dimension"},{"dragId":"gw_mea_key_fid","fid":"gw_mea_key_fid","name":"Measure names","analyticType":"dimension","semanticType":"nominal"}],"measures":[{"dragId":"gw_l9GK","fid":"item","name":"item","basename":"item","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_zf1G","fid":"sales","name":"sales","basename":"sales","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_qZUa","fid":"avg_itemPrice","name":"avg_itemPrice","basename":"avg_itemPrice","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_CfOA","fid":"avg_sales","name":"avg_sales","basename":"avg_sales","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_ln2X","fid":"甜点数量占比","name":"甜点数量占比","basename":"甜点数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_R8ar","fid":"甜点销售额占比","name":"甜点销售额占比","basename":"甜点销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_LxOt","fid":"小笼包数量占比","name":"小笼包数量占比","basename":"小笼包数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_2hEE","fid":"小笼包销售额占比","name":"小笼包销售额占比","basename":"小笼包销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_7z8V","fid":"炒饭数量占比","name":"炒饭数量占比","basename":"炒饭数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_iFHR","fid":"炒饭销售额占比","name":"炒饭销售额占比","basename":"炒饭销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_QB8r","fid":"饺子与烧卖数量占比","name":"饺子与烧卖数量占比","basename":"饺子与烧卖数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_nBcG","fid":"饺子与烧卖销售额占比","name":"饺子与烧卖销售额占比","basename":"饺子与烧卖销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_Z7Gx","fid":"馒头数量占比","name":"馒头数量占比","basename":"馒头数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_j9HL","fid":"馒头销售额占比","name":"馒头销售额占比","basename":"馒头销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_p4Vj","fid":"红豆糯米糕数量占比","name":"红豆糯米糕数量占比","basename":"红豆糯米糕数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_JCKV","fid":"红豆糯米糕销售额占比","name":"红豆糯米糕销售额占比","basename":"红豆糯米糕销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_XDGx","fid":"鸡肉小笼包数量占比","name":"鸡肉小笼包数量占比","basename":"鸡肉小笼包数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_ZKAX","fid":"鸡肉小笼包销售额占比","name":"鸡肉小笼包销售额占比","basename":"鸡肉小笼包销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_Ti0r","fid":"核桃红豆年糕数量占比","name":"核桃红豆年糕数量占比","basename":"核桃红豆年糕数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_G3aC","fid":"核桃红豆年糕销售额占比","name":"核桃红豆年糕销售额占比","basename":"核桃红豆年糕销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw__rT9","fid":"肉丝炒饭数量占比","name":"肉丝炒饭数量占比","basename":"肉丝炒饭数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_zZKA","fid":"肉丝炒饭销售额占比","name":"肉丝炒饭销售额占比","basename":"肉丝炒饭销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_blc9","fid":"虾肉猪肉水饺数量占比","name":"虾肉猪肉水饺数量占比","basename":"虾肉猪肉水饺数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_XDKn","fid":"虾肉猪肉水饺销售额占比","name":"虾肉猪肉水饺销售额占比","basename":"虾肉猪肉水饺销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_cwWB","fid":"虾肉肉丝炒饭数量占比","name":"虾肉肉丝炒饭数量占比","basename":"虾肉肉丝炒饭数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_D0wi","fid":"虾肉肉丝炒饭销售额占比","name":"虾肉肉丝炒饭销售额占比","basename":"虾肉肉丝炒饭销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_MDFX","fid":"鱼肉水饺数量占比","name":"鱼肉水饺数量占比","basename":"鱼肉水饺数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_Ted7","fid":"鱼肉水饺销售额占比","name":"鱼肉水饺销售额占比","basename":"鱼肉水饺销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_WWbd","fid":"虾肉炒饭数量占比","name":"虾肉炒饭数量占比","basename":"虾肉炒饭数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_OqzL","fid":"虾肉炒饭销售额占比","name":"虾肉炒饭销售额占比","basename":"虾肉炒饭销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_79xW","fid":"中国风蛋糕数量占比","name":"中国风蛋糕数量占比","basename":"中国风蛋糕数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_DU_K","fid":"中国风蛋糕销售额占比","name":"中国风蛋糕销售额占比","basename":"中国风蛋糕销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_xeHx","fid":"青南瓜虾肉小笼包数量占比","name":"青南瓜虾肉小笼包数量占比","basename":"青南瓜虾肉小笼包数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_JGzu","fid":"青南瓜虾肉小笼包销售额占比","name":"青南瓜虾肉小笼包销售额占比","basename":"青南瓜虾肉小笼包销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_5BCd","fid":"猪肉小笼包数量占比","name":"猪肉小笼包数量占比","basename":"猪肉小笼包数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_7Adp","fid":"猪肉小笼包销售额占比","name":"猪肉小笼包销售额占比","basename":"猪肉小笼包销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_LHU6","fid":"蔬菜猪肉水饺数量占比","name":"蔬菜猪肉水饺数量占比","basename":"蔬菜猪肉水饺数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_3_za","fid":"蔬菜猪肉水饺销售额占比","name":"蔬菜猪肉水饺销售额占比","basename":"蔬菜猪肉水饺销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_ZG9J","fid":"素馅馒头数量占比","name":"素馅馒头数量占比","basename":"素馅馒头数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_ub6t","fid":"素馅馒头销售额占比","name":"素馅馒头销售额占比","basename":"素馅馒头销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_pIv8","fid":"蔬菜猪肉馅包数量占比","name":"蔬菜猪肉馅包数量占比","basename":"蔬菜猪肉馅包数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_aAZD","fid":"蔬菜猪肉馅包销售额占比","name":"蔬菜猪肉馅包销售额占比","basename":"蔬菜猪肉馅包销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_hfnL","fid":"猪肉包子数量占比","name":"猪肉包子数量占比","basename":"猪肉包子数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_FOLp","fid":"猪肉包子销售额占比","name":"猪肉包子销售额占比","basename":"猪肉包子销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_qx_A","fid":"蟹黄猪肉小笼包数量占比","name":"蟹黄猪肉小笼包数量占比","basename":"蟹黄猪肉小笼包数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_RBCV","fid":"蟹黄猪肉小笼包销售额占比","name":"蟹黄猪肉小笼包销售额占比","basename":"蟹黄猪肉小笼包销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_r76M","fid":"素馅蒸饺数量占比","name":"素馅蒸饺数量占比","basename":"素馅蒸饺数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_eaX6","fid":"素馅蒸饺销售额占比","name":"素馅蒸饺销售额占比","basename":"素馅蒸饺销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_count_fid","fid":"gw_count_fid","name":"Row count","analyticType":"measure","semanticType":"quantitative","aggName":"sum","computed":true,"expression":{"op":"one","params":[],"as":"gw_count_fid"}},{"dragId":"gw_mea_val_fid","fid":"gw_mea_val_fid","name":"Measure values","analyticType":"measure","semanticType":"quantitative","aggName":"sum"}],"rows":[{"dragId":"gw_4JiL","fid":"sales","name":"sales","basename":"sales","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_-JzT","fid":"avg_sales","name":"avg_sales","basename":"avg_sales","analyticType":"measure","semanticType":"quantitative","aggName":"sum"}],"columns":[{"dragId":"gw_owNW","fid":"date","name":"date","basename":"date","semanticType":"nominal","analyticType":"dimension"}],"color":[],"opacity":[],"size":[],"shape":[],"radius":[],"theta":[],"longitude":[],"latitude":[],"geoId":[],"details":[],"filters":[],"text":[]},"layout":{"showActions":false,"showTableSummary":false,"stack":"stack","interactiveScale":false,"zeroScale":true,"size":{"mode":"auto","width":320,"height":200},"format":{},"geoKey":"name","resolve":{"x":false,"y":false,"color":false,"opacity":false,"shape":false,"size":false}},"visId":"gw_OjAO","name":"Chart 2"},{"config":{"defaultAggregated":true,"geoms":["line"],"coordSystem":"generic","limit":-1},"encodings":{"dimensions":[{"dragId":"gw_9Uym","fid":"date","name":"date","basename":"date","semanticType":"nominal","analyticType":"dimension"},{"dragId":"gw_0HAZ","fid":"order","name":"order","basename":"order","semanticType":"quantitative","analyticType":"dimension"},{"dragId":"gw_31Sb","fid":"今日最爱菜品","name":"今日最爱菜品","basename":"今日最爱菜品","semanticType":"nominal","analyticType":"dimension"},{"dragId":"gw_H_pJ","fid":"今日最爱品类","name":"今日最爱品类","basename":"今日最爱品类","semanticType":"nominal","analyticType":"dimension"},{"dragId":"gw_mea_key_fid","fid":"gw_mea_key_fid","name":"Measure names","analyticType":"dimension","semanticType":"nominal"}],"measures":[{"dragId":"gw_KTnR","fid":"item","name":"item","basename":"item","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_MDul","fid":"avg_item","name":"avg_item","basename":"avg_item","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_Ld2V","fid":"sales","name":"sales","basename":"sales","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_PGvv","fid":"avg_itemPrice","name":"avg_itemPrice","basename":"avg_itemPrice","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_slro","fid":"avg_sales","name":"avg_sales","basename":"avg_sales","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_CovI","fid":"甜点数量占比","name":"甜点数量占比","basename":"甜点数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_ZgGZ","fid":"甜点销售额占比","name":"甜点销售额占比","basename":"甜点销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_-Dwc","fid":"小笼包数量占比","name":"小笼包数量占比","basename":"小笼包数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_fV40","fid":"小笼包销售额占比","name":"小笼包销售额占比","basename":"小笼包销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_cg86","fid":"炒饭数量占比","name":"炒饭数量占比","basename":"炒饭数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_1VRR","fid":"炒饭销售额占比","name":"炒饭销售额占比","basename":"炒饭销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_ShLi","fid":"饺子与烧卖数量占比","name":"饺子与烧卖数量占比","basename":"饺子与烧卖数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_TogD","fid":"饺子与烧卖销售额占比","name":"饺子与烧卖销售额占比","basename":"饺子与烧卖销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_Kpju","fid":"馒头数量占比","name":"馒头数量占比","basename":"馒头数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_PjqI","fid":"馒头销售额占比","name":"馒头销售额占比","basename":"馒头销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_UeL9","fid":"红豆糯米糕数量占比","name":"红豆糯米糕数量占比","basename":"红豆糯米糕数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_zmZY","fid":"红豆糯米糕销售额占比","name":"红豆糯米糕销售额占比","basename":"红豆糯米糕销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_xBuT","fid":"鸡肉小笼包数量占比","name":"鸡肉小笼包数量占比","basename":"鸡肉小笼包数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_K08R","fid":"鸡肉小笼包销售额占比","name":"鸡肉小笼包销售额占比","basename":"鸡肉小笼包销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_8OnC","fid":"核桃红豆年糕数量占比","name":"核桃红豆年糕数量占比","basename":"核桃红豆年糕数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_kS2E","fid":"核桃红豆年糕销售额占比","name":"核桃红豆年糕销售额占比","basename":"核桃红豆年糕销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_n_W4","fid":"肉丝炒饭数量占比","name":"肉丝炒饭数量占比","basename":"肉丝炒饭数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_QK7j","fid":"肉丝炒饭销售额占比","name":"肉丝炒饭销售额占比","basename":"肉丝炒饭销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_Hm4c","fid":"虾肉猪肉水饺数量占比","name":"虾肉猪肉水饺数量占比","basename":"虾肉猪肉水饺数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_qoB8","fid":"虾肉猪肉水饺销售额占比","name":"虾肉猪肉水饺销售额占比","basename":"虾肉猪肉水饺销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_02Q_","fid":"虾肉肉丝炒饭数量占比","name":"虾肉肉丝炒饭数量占比","basename":"虾肉肉丝炒饭数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_X_Gu","fid":"虾肉肉丝炒饭销售额占比","name":"虾肉肉丝炒饭销售额占比","basename":"虾肉肉丝炒饭销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw__HS6","fid":"鱼肉水饺数量占比","name":"鱼肉水饺数量占比","basename":"鱼肉水饺数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_AC2D","fid":"鱼肉水饺销售额占比","name":"鱼肉水饺销售额占比","basename":"鱼肉水饺销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_paUO","fid":"虾肉炒饭数量占比","name":"虾肉炒饭数量占比","basename":"虾肉炒饭数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_D9-1","fid":"虾肉炒饭销售额占比","name":"虾肉炒饭销售额占比","basename":"虾肉炒饭销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_4r_n","fid":"中国风蛋糕数量占比","name":"中国风蛋糕数量占比","basename":"中国风蛋糕数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_0M8q","fid":"中国风蛋糕销售额占比","name":"中国风蛋糕销售额占比","basename":"中国风蛋糕销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_a_fX","fid":"青南瓜虾肉小笼包数量占比","name":"青南瓜虾肉小笼包数量占比","basename":"青南瓜虾肉小笼包数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_lbiC","fid":"青南瓜虾肉小笼包销售额占比","name":"青南瓜虾肉小笼包销售额占比","basename":"青南瓜虾肉小笼包销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_9upB","fid":"猪肉小笼包数量占比","name":"猪肉小笼包数量占比","basename":"猪肉小笼包数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_rkcv","fid":"猪肉小笼包销售额占比","name":"猪肉小笼包销售额占比","basename":"猪肉小笼包销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_02sY","fid":"蔬菜猪肉水饺数量占比","name":"蔬菜猪肉水饺数量占比","basename":"蔬菜猪肉水饺数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_hyP1","fid":"蔬菜猪肉水饺销售额占比","name":"蔬菜猪肉水饺销售额占比","basename":"蔬菜猪肉水饺销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_j8dH","fid":"素馅馒头数量占比","name":"素馅馒头数量占比","basename":"素馅馒头数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_zrsn","fid":"素馅馒头销售额占比","name":"素馅馒头销售额占比","basename":"素馅馒头销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_Ir5Y","fid":"蔬菜猪肉馅包数量占比","name":"蔬菜猪肉馅包数量占比","basename":"蔬菜猪肉馅包数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_hvLY","fid":"蔬菜猪肉馅包销售额占比","name":"蔬菜猪肉馅包销售额占比","basename":"蔬菜猪肉馅包销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_tyKR","fid":"猪肉包子数量占比","name":"猪肉包子数量占比","basename":"猪肉包子数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_8Quu","fid":"猪肉包子销售额占比","name":"猪肉包子销售额占比","basename":"猪肉包子销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_2q-_","fid":"蟹黄猪肉小笼包数量占比","name":"蟹黄猪肉小笼包数量占比","basename":"蟹黄猪肉小笼包数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_6J02","fid":"蟹黄猪肉小笼包销售额占比","name":"蟹黄猪肉小笼包销售额占比","basename":"蟹黄猪肉小笼包销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_pVCX","fid":"素馅蒸饺数量占比","name":"素馅蒸饺数量占比","basename":"素馅蒸饺数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_oLL_","fid":"素馅蒸饺销售额占比","name":"素馅蒸饺销售额占比","basename":"素馅蒸饺销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_count_fid","fid":"gw_count_fid","name":"Row count","analyticType":"measure","semanticType":"quantitative","aggName":"sum","computed":true,"expression":{"op":"one","params":[],"as":"gw_count_fid"}},{"dragId":"gw_mea_val_fid","fid":"gw_mea_val_fid","name":"Measure values","analyticType":"measure","semanticType":"quantitative","aggName":"sum"}],"rows":[{"dragId":"gw_3l8r","fid":"小笼包数量占比","name":"小笼包数量占比","basename":"小笼包数量占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_rpbr","fid":"小笼包销售额占比","name":"小笼包销售额占比","basename":"小笼包销售额占比","analyticType":"measure","semanticType":"quantitative","aggName":"sum"}],"columns":[{"dragId":"gw_wL1i","fid":"date","name":"date","basename":"date","semanticType":"nominal","analyticType":"dimension"}],"color":[],"opacity":[],"size":[],"shape":[],"radius":[],"theta":[],"longitude":[],"latitude":[],"geoId":[],"details":[],"filters":[],"text":[]},"layout":{"showActions":false,"showTableSummary":false,"stack":"stack","interactiveScale":false,"zeroScale":true,"size":{"mode":"auto","width":320,"height":200},"format":{},"geoKey":"name","resolve":{"x":false,"y":false,"color":false,"opacity":false,"shape":false,"size":false}},"visId":"gw_EaLU","name":"Chart 4"}],"chart_map":{},"workflow_list":[{"workflow":[{"type":"view","query":[{"op":"aggregate","groupBy":["date"],"measures":[{"field":"order","asFieldKey":"order"},{"field":"item","agg":"sum","asFieldKey":"item_sum"},{"field":"avg_item","agg":"sum","asFieldKey":"avg_item_sum"},{"field":"avg_itemPrice","agg":"sum","asFieldKey":"avg_itemPrice_sum"}]}]}]},{"workflow":[{"type":"view","query":[{"op":"aggregate","groupBy":["date"],"measures":[{"field":"sales","agg":"sum","asFieldKey":"sales_sum"},{"field":"avg_sales","agg":"sum","asFieldKey":"avg_sales_sum"}]}]}]},{"workflow":[{"type":"view","query":[{"op":"aggregate","groupBy":["date"],"measures":[{"field":"小笼包数量占比","agg":"sum","asFieldKey":"小笼包数量占比_sum"},{"field":"小笼包销售额占比","agg":"sum","asFieldKey":"小笼包销售额占比_sum"}]}]}]}],"version":"0.4.6"}"""
# pyg.walk(df, spec=vis_spec)
# 这个数据可视化可以再进一步了解

# 计算最小样本量
# 计算均值、标准差
from statsmodels.stats.power import NormalIndPower

df_ctrl = df[df['date'] < '2023-04-01']
df_test = df[df['date'] >= '2023-04-01']

a_list = []
b_list = []

for columns in ['order', 'item', 'avg_item', 'sales', 'avg_itemPrice', 'avg_sales']:
    # 将数据按实验组分为A组和B组
    group_a = df_test[columns]
    group_b = df_ctrl[columns]

    ma = group_a.mean()
    va = group_a.std()

    mb = group_b.mean()
    vb = group_b.std()

    a_list.append([ma, va])
    b_list.append([mb, vb])

a_list = pd.DataFrame(a_list, index=['order', 'item', 'avg_item', 'sales', 'avg_itemPrice', 'avg_sales'],
                      columns=['mean', 'std'])
b_list = pd.DataFrame(b_list, index=['order', 'item', 'avg_item', 'sales', 'avg_itemPrice', 'avg_sales'],
                      columns=['mean', 'std'])

# 计算最小样本量:足够大的样本量，样本量还得有代表性（这需要先行进行控制）
effect_size = 0
for columns in ['order', 'item', 'avg_item', 'sales', 'avg_itemPrice', 'avg_sales']:
    std = df[columns].std()
    temp = abs(a_list[a_list.index == columns]['mean'] - b_list[a_list.index == columns]['mean'])
    tt = temp.values / std
    if effect_size < tt:
        effect_size = tt

effect_size = effect_size[0]

num = NormalIndPower().solve_power(
    effect_size=effect_size,
    nobs1=None,
    alpha=0.05,
    power=0.8,
    ratio=1,
    alternative='two-sided',
)
#在每组中至少需要有大约4.47（num）个观测值

#检验
# 常用的检验方法：
# Z检验：当样本量较大（大于30）且总体标准差已知时，使用z检验。
# t检验：当样本量较小（小于30）或总体标准差未知时，使用t检验。

import scipy.stats as stats

df_ctrl = df[df['date'] < '2023-04-01']
df_test = df[df['date'] >= '2023-04-01']

t_list = []

for columns in ['order', 'item', 'avg_item', 'sales', 'avg_itemPrice', 'avg_sales']:
    # 将数据按实验组分为A组和B组
    group_a = df_test[columns]
    group_b = df_ctrl[columns]

    # 执行独立样本t检验
    t_statistic, t_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    t_list.append([t_statistic, round(t_value, 7)])

t_list = pd.DataFrame(t_list, index=['order', 'item', 'avg_item', 'sales', 'avg_itemPrice', 'avg_sales'],
                      columns=['t_statistic', 't_value'])
# print(t_list)

# p值小于 0.05， 则拒绝原假设，认为新功能上线后指标提高了:错误不超过5%

p_list = []
columns_list = list(df.columns)[list(df.columns).index('甜点数量占比'):]
for columns in columns_list:
    # 将数据按实验组分为A组和B组
    group_a = df[df['date'] >= '2023-04-01'][columns]
    group_b = df[df['date'] < '2023-04-01'][columns]

    # 执行独立样本t检验
    t_statistic, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)

    p_list.append([t_statistic, round(p_value, 7)])

p_df = pd.DataFrame(p_list, index=columns_list, columns=['t_statistic', 'p_value'])
# p_df

import statsmodels.stats.weightstats as sw

z_list = []

for columns in ['order', 'item', 'avg_item', 'sales', 'avg_itemPrice', 'avg_sales']:
    # 将数据按实验组分为A组和B组
    group_a = df_test[columns]
    group_b = df_ctrl[columns]

    z_statistic, z_value = sw.ztest(group_a, group_b, value=0)
    z_list.append([z_statistic, round(z_value, 7)])

z_list = pd.DataFrame(z_list, index=['order', 'item', 'avg_item', 'sales', 'avg_itemPrice', 'avg_sales'],
                      columns=['z_statistic', 'z_value'])
print(z_list)
