from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
# 读取Excel数据
data = pd.read_excel("D:\\CUMCM2023Problems\\C题\\附件2（销售情况）.xlsx")

data['销售日期'] = pd.to_datetime(data['销售日期'])
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 过滤出2023年6月的销售数据
sales_june_2023 = data[(data['销售日期'].dt.year == 2023) & (data['销售日期'].dt.month == 6)]

# 获取大类列表
category_list = sales_june_2023['大类'].unique()
# 遍历每个大类，进行白噪声检验
for category in category_list:
    category_data = sales_june_2023[sales_june_2023['大类'] == category]['销售单价(元/千克)']

    # Ljung-Box检验
    lb_test_results = acorr_ljungbox(category_data, lags=10, return_df=True)
    p_values = lb_test_results['lb_pvalue']
    max_p_value = p_values.max()
    print("Ljung-Box检验 - 大类:", category)
    print("最大P值:", max_p_value)

    # Durbin-Watson检验
    dw_test_statistic, dw_p_value = durbin_watson(category_data)
    print("Durbin-Watson检验 - 大类:", category)
    print("统计量:", dw_test_statistic)
    print("P值:", dw_p_value)
    print("------------------------------------")
