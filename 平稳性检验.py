import pandas as pd
from statsmodels.tsa.stattools import adfuller

# 读取 Excel 文件
data = pd.read_excel("D:\\CUMCM2023Problems\\C题\\6月数据.xlsx")

# 将销售日期列转换为日期时间类型
data["销售日期"] = pd.to_datetime(data["销售日期"])

# 筛选出2023年6月份的数据
june_2023_data = data[(data["销售日期"].dt.year == 2023) & (data["销售日期"].dt.month == 6)]

# 对销量进行平稳性检验
quantity_data = june_2023_data["销量(千克)"]
quantity_result = adfuller(quantity_data)


# 对销售单价进行平稳性检验
price_data = june_2023_data["销售单价(元/千克)"]
price_result = adfuller(price_data)


# 递归进行1到10阶差分并进行平稳性检验
for i in range(1, 11):
    print(f"进行 {i} 阶差分的平稳性检验：")
    quantity_diff = quantity_data.diff(i).dropna()
    price_diff = price_data.diff(i).dropna()

    quantity_diff_result = adfuller(quantity_diff)
    print(f"销量（千克）{i} 阶差分的平稳性检验结果：")
    print(f"ADF Statistic: {quantity_diff_result[0]}")
    print(f"p-value: {quantity_diff_result[1]}")
    print("Critical Values:")
    for key, value in quantity_diff_result[4].items():
        print(f"\t{key}: {value}")
    print("-----------------------------------------")

    price_diff_result = adfuller(price_diff)
    print(f"销售单价（元/千克）{i} 阶差分的平稳性检验结果：")
    print(f"ADF Statistic: {price_diff_result[0]}")
    print(f"p-value: {price_diff_result[1]}")
    print("Critical Values:")
    for key, value in price_diff_result[4].items():
        print(f"\t{key}: {value}")
    print("-----------------------------------------")
