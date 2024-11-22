import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

# 读取Excel数据
data = pd.read_excel("D:\\CUMCM2023Problems\\C题\\6月数据.xlsx")

# 将数据按照大类分组
grouped_data = data.groupby('大类')

# 定义显著水平
significance_levels = [0.05, 0.1]

# 遍历每个大类
for group_name, group_data in grouped_data:
    # 选择需要的列
    group_data = group_data[['销量(千克)', '销售单价(元/千克)']]

    # 进行格兰杰因果分析
    result = grangercausalitytests(group_data, maxlag=1)

    # 打印格兰杰因果分析的结果
    print(f"大类 {group_name} 的格兰杰因果分析结果:")
    p_value = result[1][0]['ssr_chi2test'][1]
    for significance_level in significance_levels:
        if p_value < significance_level:
            causal_relationship = "存在因果关系"
        else:
            causal_relationship = "不存在因果关系"

        print(f"Lag 1, p-value = {p_value}, 在显著水平 {significance_level} 下，{causal_relationship}")
    print()

