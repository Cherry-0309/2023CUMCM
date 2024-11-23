import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the Excel file
data = pd.read_excel("D:\\CUMCM2023Problems\\C题\\6月数据.xlsx",sheet_name='Sheet1')

# Convert the "销售日期" column to datetime format
data['销售日期'] = pd.to_datetime(data['销售日期'])

# Filter the data for sales in June 2023
data_june = data[(data['销售日期'] >= pd.Timestamp(2023, 6, 1)) & (data['销售日期'] <= pd.Timestamp(2023, 6, 30))]

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# Define a custom color palette with darker colors
color_palette = ['#508EA4', '#406E7E', '#345465', '#28444D', '#1C3337', '#102228']

# Split the data by category
grouped_data = data_june.groupby('大类')

# Create a figure with six subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
fig.suptitle('各单品类2023年6月销售量时序图', fontsize=16)

# Iterate over each category and plot the time series graph for sales in June 2023
for (i, (category, df)), ax in zip(enumerate(grouped_data), axes.flatten()):
    sns.lineplot(x='销售日期', y='销量(千克)', data=df, ax=ax, color=color_palette[i], linewidth=2)
    ax.set_title(category, color=color_palette[i])
    ax.set_xlabel('日期')
    ax.set_ylabel('销量(千克)')
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))  # Set the maximum number of x-axis ticks to 6
    ax.spines['right'].set_visible(False)  # Hide right spine
    ax.spines['top'].set_visible(False)  # Hide top spine

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.5, wspace=0.3)

# Remove the grid lines
sns.despine()

# Display the graph
plt.show()
