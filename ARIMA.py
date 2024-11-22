import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 加载数据
file_path = "D:\\CUMCM2023Problems\\C题\\6月销售总量.xlsx"
sales_data = pd.read_excel(file_path)
#处理中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']


# 设置滞后数和差分阶数
lags = 5
diff_order = 1

# 提取销量数据列
sales = sales_data['销量(千克)']

# 差分操作
sales_diff = sales.diff(diff_order).dropna()

# 创建 ARIMA 模型
model = ARIMA(sales_diff, order=(lags, diff_order, 4))
model_fit = model.fit()

# 模型预测
forecast = model_fit.predict(start=len(sales_diff), end=len(sales_diff) + 6)

# 还原差分
forecast_restored = sales.iloc[-1] + np.cumsum(forecast)

# 指数平滑
alpha = 0.2
smoothed_forecast = forecast_restored.ewm(alpha=alpha).mean()

# 绘制原始数据和平滑预测结果的对比图
plt.figure(figsize=(10, 6))
plt.plot(sales, label='实际销量')
plt.plot(smoothed_forecast, label='预测销量')
plt.xlabel('时间')
plt.ylabel('销量(千克)')
plt.title('ARIMA (11, 4, 2) 平滑预测模型')
plt.legend()
plt.show()

# 输出预测结果
print("平滑预测结果：")
print(smoothed_forecast)
