import pandas as pd
import numpy as np
from deap import creator, base, tools, algorithms

# 读取 Excel 文件
filename = r"D:\CUMCM2023Problems\C题\问题三数据处理.xlsx"
sheet_name = "Sheet1"


# 读取数据
data = pd.read_excel(filename, sheet_name=sheet_name)
# 提取所需列
sales_data = data[['日期', '单品编码', '日总销量', '成本', '销售单价(元/千克)']].copy()

# 设置参数
target_num_items = 31  # 目标可售单品总数
min_display_amount = 2.5  # 最小陈列量


# 定义适应度函数
def calculate_fitness(individual):
    selected_items = sales_data.loc[individual == 1]  # 选取被选中的单品数据
    total_sales = selected_items['日总销量'].sum()
    total_cost = selected_items['成本'].sum()
    return total_sales - total_cost,  # 返回盈利额作为目标


# 定义问题的类型（最大化盈利额）
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

# 创建遗传算法的工具箱
toolbox = base.Toolbox()

# 定义个体编码方式
toolbox.register("attr_bool", np.random.randint, 0, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(sales_data))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 定义评估函数
toolbox.register("evaluate", calculate_fitness)

# 定义交叉操作
toolbox.register("mate", tools.cxOnePoint)

# 定义变异操作
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# 定义选择操作
toolbox.register("select", tools.selTournament, tournsize=3)

# 定义进化算法的参数
population_size = 100
crossover_rate = 0.8
mutation_rate = 0.2
num_generations = 50

# 创建初始种群
population = toolbox.population(n=population_size)

# 运行遗传算法
for generation in range(num_generations):
    # 计算每个个体的适应度值
    fitness_values = [toolbox.evaluate(individual) for individual in population]

    # 更新每个个体的适应度值
    for individual, fitness in zip(population, fitness_values):
        individual.fitness.values = fitness

    # 选择下一代个体
    offspring = toolbox.select(population, len(population))

    # 复制选出的个体
    offspring = list(map(toolbox.clone, offspring))

    # 对选出的个体进行交叉操作和变异操作
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if np.random.rand() < crossover_rate:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if np.random.rand() < mutation_rate:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # 计算新一代个体的适应度值
    invalid_individuals = [individual for individual in offspring if not individual.fitness.valid]
    fitness_values = [toolbox.evaluate(individual) for individual in invalid_individuals]

    # 更新新一代个体的适应度值
    for individual, fitness in zip(invalid_individuals, fitness_values):
        individual.fitness.values = fitness

    # 替换种群中的个体
    population[:] = offspring

# 获取优化后的最佳个体
best_individual = tools.selBest(population, 1)[0]

# 输出结果
selected_items = sales_data.loc[best_individual == 1]
selected_items = selected_items[selected_items['日总销量'] >= min_display_amount]  # 筛选满足最小陈列量的单品
selected_items.to_excel('selected_items.xlsx', index=False)

