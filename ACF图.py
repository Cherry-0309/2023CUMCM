import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# 读取Excel数据
data = pd.read_excel("D:\\CUMCM2023Problems\\C题\\6月数据.xlsx")

data['销售日期'] = pd.to_datetime(data['销售日期'])
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 过滤出2023年6月的销售数据
sales_june_2023 = data[(data['销售日期'].dt.year == 2023) & (data['销售日期'].dt.month == 6)]

# 获取大类列表
category_list = sales_june_2023['大类'].unique()

# 计算子图的行数和列数
num_rows = (len(category_list) - 1) // 2 + 1
num_cols = min(len(category_list), 2)

# 创建一个大图，并设置子图布局
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
fig.tight_layout(pad=5)

# 遍历每个大类，绘制ACF图像
for i, category in enumerate(category_list):
    row_idx = i // num_cols
    col_idx = i % num_cols

    # 获取当前子图的坐标轴
    if num_rows > 1:
        ax = axes[row_idx, col_idx]
    else:
        ax = axes[col_idx]

    category_data = sales_june_2023[sales_june_2023['大类'] == category]['销售单价(元/千克)']

    # 绘制ACF图像并设置线条样式
    plot_acf(category_data, lags=30, ax=ax, linewidth=1.5, color='steelblue')
    ax.set_title(category + ' - 自相关图（ACF）')

plt.show()
