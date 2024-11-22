import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 从Excel文件加载数据
file_path = "D:\\CUMCM2023Problems\\C题\\6月销售总量.xlsx"
sales_data = pd.read_excel(file_path)

# 将"销售日期"列转换为日期类型
sales_data['销售日期'] = pd.to_datetime(sales_data['销售日期'])

# 设置滞后数
lags = 11

# 提取销量数据列
sales = sales_data['销量(千克)']

# 创建图像
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

# 绘制 ACF 图像
plot_acf(sales, lags=lags, ax=ax[0], linewidth=1, color='steelblue')
ax[0].set_title('ACF')

# 绘制 PACF 图像
plot_pacf(sales, lags=lags, ax=ax[1], linewidth=1, color='steelblue')
ax[1].set_title('PACF')

# 调整子图之间的间距
fig.tight_layout()

# 显示图像
plt.show()
