import pandas as pd

# 读取数据
df_sales = pd.read_excel("D:/CUMCM2023Problems/C题/附件2（销售情况）.xlsx")
df_wholesale = pd.read_excel("D:/CUMCM2023Problems/C题/附件3（批发）.xlsx")
df_loss = pd.read_excel("D:/CUMCM2023Problems/C题/附件4（损耗率）.xlsx")

# 合并数据
df_merged = pd.merge(df_sales, df_wholesale, on='单品编码', how='left')
df_merged = pd.merge(df_merged, df_loss, on='单品编码', how='left')

# 选择并重命名列
df_merged = df_merged[['单品编码', '销量(千克)', '批发价格(元/千克)', '损耗率(%)']]
df_merged = df_merged.rename(columns={'销量(千克)': '销量', '批发价格(元/千克)': '批发价格', '损耗率(%)': '损耗率'})

# 保存结果到Excel文件
df_merged.to_excel('D:/CUMCM2023Problems/C题/表1.xlsx', index=False)
