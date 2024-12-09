"""
author: Yiting Chen
description: using SVR to implement stock prediction, using Genetic Algorithm to select best parameters of SVR
"""

from DataSource import YahooDataSet
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score

#SVR
from sklearn.svm import SVR
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

#deap
from deap import base, creator, tools, algorithms

"""
最优参数模型
"""
def Best_SVR_model():
    dataset = YahooDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()

    def mixed_kernel(X, Y, a=0.5, b=0.5, rbf_gamma=0.1, poly_gamma=0.1, coef0=1, degree=10):
        # RBF 核
        rbf = rbf_kernel(X, Y, gamma=rbf_gamma)
        # 多项式核
        poly = polynomial_kernel(X, Y, gamma=poly_gamma, coef0=coef0, degree=degree)
        # 混合核
        return a * rbf + b * poly

    # 初始化SVR模型，epsilon是惩罚平面的距离，超过这个距离就会被penalize
    #k(x, x')= exp(-gamma ||x - x'||^2), 较小的 gamma 值：核函数的范围更大（对距离更不敏感）。每个数据点的影响范围更宽广。模型更倾向于低复杂度，可能会欠拟合。

     #最佳参数：a=0.18491860149827433 ,C=67.66251338814207, epsilon=0.033488499218959696, rbf_gamma=0.0008777144162348549, poly_gamma=0.0004245042477603652, coef0=2.997984163089049, degree=3
    best_a, best_C, best_epsilon = 0.185, 67.66, 0.0335
    best_rbf_gamma = 0.00087
    best_poly_gamma, best_poly_coef0, best_poly_degree = 0.000424, 3, 3
    # lambda 为匿名函数，等于创造一个函数func(X,Y)，不过用匿名函数可以在程序运行时动态调整传入参数的大小 e.g.a，b
    svr = SVR(kernel=lambda X, Y: mixed_kernel(X, Y, a=best_a, b=1-best_a, rbf_gamma=best_rbf_gamma, poly_gamma=best_poly_gamma, coef0=best_poly_coef0, degree=best_poly_degree),
              C=best_C,
              epsilon=best_epsilon
              )

    # 训练模型
    svr.fit(X_train, y_train.ravel())

    # 预测
    y_pred_test = svr.predict(X_test)
    y_pred_train = svr.predict(X_train)
    print(f'mean_squared_error on test set = {mean_squared_error(y_train, y_pred_train)}')
    print(f'mean_squared_error on test set = {mean_squared_error(y_test, y_pred_test)}')

    df_data = dataset.data
    n_rows = len(df_data)
    train_indices = df_data.index[0: int(0.8 * n_rows)]
    test_indices = df_data.index[int(0.8 * n_rows):]

    # 设置画布大小
    plt.figure(figsize=(12, 6))

    #绘制prediction on training set
    # 预测值折线图
    plt.plot(
        train_indices,
        y_pred_train,
        label='Prediction',
        color='blue',
        linewidth=1.5,  # 调整线条宽度
        alpha=0.8  # 设置透明度
    )

    # 真实值折线图
    plt.plot(
        train_indices,
        y_train,
        label='Truth',
        color='red',
        linewidth=1.5,  # 调整线条宽度
        alpha=0.8  # 设置透明度
    )

    # 设置标题和轴标签
    plt.title('SVR Prediction vs Truth on Training Set', fontsize=16)  # 标题和字体大小
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Price', fontsize=12)

    # 设置图例
    plt.legend(fontsize=12)

    # 显示网格线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 显示图形
    plt.tight_layout()
    plt.show()

    # 绘制prediction on test set
    # 设置画布大小
    plt.figure(figsize=(12, 6))

    # 预测值折线图
    plt.plot(
        test_indices,
        y_pred_test,
        label='Prediction',
        color='blue',
        linewidth=1.5,  # 调整线条宽度
        alpha=0.8  # 设置透明度
    )

    # 真实值折线图
    plt.plot(
        test_indices,
        y_test,
        label='Truth',
        color='red',
        linewidth=1.5,  # 调整线条宽度
        alpha=0.8  # 设置透明度
    )

    # 设置标题和轴标签
    plt.title('SVR Prediction vs Truth on Test Set', fontsize=16)  # 标题和字体大小
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Price', fontsize=12)

    # 设置图例
    plt.legend(fontsize=12)

    # 显示网格线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 显示图形
    plt.tight_layout()
    plt.show()

"""
一次性根据GA获取所有最优参数，由于SVR的kernel matrix计算十分缓慢，所以算的太慢了，需要更强的性能或者GPU加速。
所以替代的可以依次单独跑下面提供的方法，单独获取各部分最优参数拼凑起来（局部最优并不一定代表全局最优），但运行时间在可接受范围内了
"""
def GA():
    dataset = YahooDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()

    #自定义混合kernel
    def mixed_kernel(X, Y, a=0.5, b=0.5, rbf_gamma=0.1, poly_gamma=0.1, coef0=1, degree=3):
        # RBF 核
        rbf = rbf_kernel(X, Y, gamma=rbf_gamma)
        # 多项式核
        poly = polynomial_kernel(X, Y, gamma=poly_gamma, coef0=coef0, degree=degree)
        # 混合核
        return a * rbf + b * poly

    # 定义遗传算法的优化目标函数
    def fitness(individual):
        a, C, epsilon, rbf_gamma, poly_gamma, coef0, degree = individual
        #print(f"calculate fitness：a={a} ,C={C}, epsilon={epsilon}, gamma={gamma}, coef0={coef0}, degree={degree}")
        #model = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)
        # lambda 为匿名函数，等于创造一个函数func(X,Y)，不过用匿名函数可以在程序运行时动态调整传入参数的大小 e.g.a，b
        model = SVR(kernel=lambda X, Y: mixed_kernel(X, Y, a=a, b=1-a, rbf_gamma=rbf_gamma, poly_gamma=poly_gamma, coef0=coef0, degree=degree),
                  C=C,
                  epsilon=epsilon
                  )
        # 使用交叉验证评估模型性能(太慢了)
        #scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
        # return np.mean(scores),  # 目标是最大化负的 MSE (最小化 MSE)

        #快一些，就fit一次
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return -mse,


    # 定义 GA 的搜索范围
    a_range = (0,1) #b=1-a

    C_range = (0.1, 100)  # C 的范围
    epsilon_range = (0.01, 1)  # epsilon 的范围
    #rbf kernel超参数
    rbf_gamma_range = (0.00001, 0.001)  # rbf gamma 的范围
    #polu kernel超参数
    poly_gamma_range = (0.00001, 0.001)  # poly gamma 的范围
    coef0_range = (-1, 3)
    degree_range = (2, 5)

    # 修正函数，确保参数在范围内(由于遗传算法在mate、mutate之后会溢出范围，需要修正)
    def repair_bounds(individual, param_ranges):
        length = len(param_ranges)
        for i, (min_val, max_val) in enumerate(param_ranges):
            individual[i] = np.clip(individual[i], min_val, max_val)
            if i == length - 1: #this parameter is poly kernel degree, it must be an integer
                individual[i] = int(individual[i])
        return individual

    # 包装交叉操作，修正交叉后的参数范围
    def constrained_cxBlend(ind1, ind2, alpha=0.5):
        # 混合交叉，可以直接看源码，就几行。就是交叉后offspring[i]=(1. - gamma) * ind1[i] + gamma * ind2[i],gama是基于一个随机数产生
        #alpha=0.5，gama \in [-1,0.5),所以交叉后的值不会偏离父值太远
        tools.cxBlend(ind1, ind2, alpha)
        param_ranges = [a_range , C_range, epsilon_range, rbf_gamma_range, poly_gamma_range, coef0_range, degree_range]
        repair_bounds(ind1, param_ranges)
        repair_bounds(ind2, param_ranges)
        return ind1, ind2

    # 包装变异操作，修正变异后的参数范围
    def constrained_mutate(individual, mu=0, sigma=0.1, indpb=0.2):
        tools.mutGaussian(individual, mu, sigma, indpb) #高斯变异，变异离原来的值不会差太远
        param_ranges = [a_range ,C_range, epsilon_range, rbf_gamma_range, poly_gamma_range, coef0_range, degree_range]
        repair_bounds(individual, param_ranges)
        return individual,

    # 创建 DEAP 遗传算法工具箱
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # 最大化目标
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    # 定义个体及其属性
    toolbox.register("attr_a", np.random.uniform, *a_range)
    toolbox.register("attr_C", np.random.uniform, *C_range) #调用np.random.uniform(C_range)初始化attr_C
    toolbox.register("attr_epsilon", np.random.uniform, *epsilon_range)
    toolbox.register("attr_rbf_gamma", np.random.uniform, *rbf_gamma_range)
    toolbox.register("attr_poly_gamma", np.random.uniform, *poly_gamma_range)
    toolbox.register("attr_coef0", np.random.uniform, *coef0_range)
    toolbox.register("attr_degree", np.random.randint, *degree_range)


    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_a, toolbox.attr_C, toolbox.attr_epsilon, toolbox.attr_rbf_gamma, toolbox.attr_poly_gamma, toolbox.attr_coef0, toolbox.attr_degree), n=1) #n=1表示生成一次(toolbox.attr_C, toolbox.attr_epsilon, toolbox.attr_gamma)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 注册遗传算法操作
    toolbox.register("select", tools.selTournament, tournsize=3) #竞标赛选择, 在eaSimple中默认k=len(population)，即选出len(population)个用于交配
    toolbox.register("mate", constrained_cxBlend)  # 混合交叉 eaSimple中操作就是让每相邻两个parent进行交配（概率行为）
    toolbox.register("mutate", constrained_mutate)  # 高斯变异 eaSimple也是遍历所有offspring，让其变异（概率行为）
    toolbox.register("evaluate", fitness)

    # 设置 GA 参数
    population = toolbox.population(n=50)  # 种群大小
    n_generations = 20  # 迭代次数

    # 运行遗传算法 cxpb: 交叉(mate/crossover)概率。mutpb: 变异概率。环境选择采用全替换策略
    #输出中，nevals小于population是由于mutate和mate是概率行为，有的individual没有经过mate和mutate，就不会评估
    result_population, _ = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=n_generations,
                                               verbose=True)

    # 找到最优个体
    best_individual = tools.selBest(result_population, k=1)[0]
    best_a, best_C, best_epsilon, best_rbf_gamma, best_poly_gamma, best_coef0, best_degree = best_individual
    print(f"最佳参数：a={best_a} ,C={best_C}, epsilon={best_epsilon}, rbf_gamma={best_rbf_gamma}, poly_gamma={best_poly_gamma},, coef0={best_coef0}, degree={best_degree}")

    # 使用最佳参数训练最终模型
    final_model = SVR(kernel=lambda X, Y: mixed_kernel(X, Y, a=best_a, b=1 - best_a, rbf_gamma=best_rbf_gamma, poly_gamma=best_poly_gamma, coef0=best_coef0, degree=best_degree),
                C=best_C,
                epsilon=best_epsilon
                )
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    # 输出最终模型的性能
    mse = mean_squared_error(y_test, y_pred)
    print(f"测试集均方误差 (MSE): {mse}")

"""
单独获取RBF核的最优参数
"""
def getBestRBFParametersByGA():
    dataset = YahooDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()

    # 定义遗传算法的优化目标函数
    def fitness(individual):
        C, epsilon, gamma = individual
        model = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)
        # 使用交叉验证评估模型性能(太慢了)
        scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
        return np.mean(scores),  # 目标是最大化负的 MSE (最小化 MSE)


    # 定义 GA 的搜索范围
    C_range = (0.1, 100)  # C 的范围
    epsilon_range = (0.01, 1)  # epsilon 的范围
    # rbf kernel超参数
    gamma_range = (0.00007, 0.0005)  # gamma 的范围


    # 修正函数，确保参数在范围内(由于遗传算法在mate、mutate之后会溢出范围，需要修正)
    def repair_bounds(individual, param_ranges):
        for i, (min_val, max_val) in enumerate(param_ranges):
            individual[i] = np.clip(individual[i], min_val, max_val)
        return individual

    # 包装交叉操作，修正交叉后的参数范围
    def constrained_cxBlend(ind1, ind2, alpha=0.5):
        # 混合交叉，可以直接看源码，就几行。就是交叉后offspring[i]=(1. - gamma) * ind1[i] + gamma * ind2[i],gama是基于一个随机数产生
        # alpha=0.5，gama \in [-1,0.5),所以交叉后的值不会偏离父值太远
        tools.cxBlend(ind1, ind2, alpha)
        param_ranges = [C_range, epsilon_range, gamma_range]
        repair_bounds(ind1, param_ranges)
        repair_bounds(ind2, param_ranges)
        return ind1, ind2

    # 包装变异操作，修正变异后的参数范围
    def constrained_mutate(individual, mu=0, sigma=0.1, indpb=0.2):
        tools.mutGaussian(individual, mu, sigma, indpb)  # 高斯变异，变异离原来的值不会差太远
        param_ranges = [C_range, epsilon_range, gamma_range]
        repair_bounds(individual, param_ranges)
        return individual,

    # 创建 DEAP 遗传算法工具箱
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # 最大化目标
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    # 定义个体及其属性
    toolbox.register("attr_C", np.random.uniform, *C_range)  # 调用np.random.uniform(C_range)初始化attr_C
    toolbox.register("attr_epsilon", np.random.uniform, *epsilon_range)
    toolbox.register("attr_gamma", np.random.uniform, *gamma_range)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_C, toolbox.attr_epsilon, toolbox.attr_gamma), n=1)  # n=1表示生成一次(toolbox.attr_C, toolbox.attr_epsilon, toolbox.attr_gamma)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 注册遗传算法操作
    toolbox.register("select", tools.selTournament,
                     tournsize=3)  # 竞标赛选择, 在eaSimple中默认k=len(population)，即选出len(population)个用于交配
    toolbox.register("mate", constrained_cxBlend)  # 混合交叉 eaSimple中操作就是让每相邻两个parent进行交配（概率行为）
    toolbox.register("mutate", constrained_mutate)  # 高斯变异 eaSimple也是遍历所有offspring，让其变异（概率行为）
    toolbox.register("evaluate", fitness)

    # 设置 GA 参数
    population = toolbox.population(n=50)  # 种群大小
    n_generations = 20  # 迭代次数

    # 运行遗传算法 cxpb: 交叉(mate/crossover)概率。mutpb: 变异概率。环境选择采用全替换策略
    # 输出中，nevals小于population是由于mutate和mate是概率行为，有的individual没有经过mate和mutate，就不会评估
    result_population, _ = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=n_generations,
                                               verbose=True)

    # 找到最优个体
    best_individual = tools.selBest(result_population, k=1)[0]
    best_C, best_epsilon, best_gamma = best_individual
    print(
        f"最佳参数：C={best_C}, epsilon={best_epsilon}, gamma={best_gamma}")

    # 使用最佳参数训练最终模型
    final_model = SVR(kernel='rbf', C=best_C, epsilon=best_epsilon, gamma=best_gamma)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    # 输出最终模型的性能
    mse = mean_squared_error(y_test, y_pred)
    print(f"测试集均方误差 (MSE): {mse}")
    #result: 最佳参数：C=100.0, epsilon=0.011996595335995544, gamma=0.00014933522401196136

"""
单独获取Poly核的最优参数
"""
def getBestPolyParametersByGA():
    dataset = YahooDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()

    # 定义遗传算法的优化目标函数
    def fitness(individual):
        C, epsilon, gamma, coef0, degree = individual
        model = SVR(kernel='poly', C=C, gamma=gamma,epsilon=epsilon, coef0=coef0, degree=degree)
        # 使用交叉验证评估模型性能(太慢了)
        #scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
        #return np.mean(scores),  # 目标是最大化负的 MSE (最小化 MSE)
        # 快一些，就fit一次
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return -mse,

    # 定义 GA 的搜索范围

    C_range = (0.1, 100)  # C 的范围
    epsilon_range = (0.01, 1)  # epsilon 的范围
    # polu kernel超参数
    gamma_range = (0.00005, 0.001)  # gamma 的范围
    coef0_range = (-1, 3)
    degree_range = (2, 5)

    # 修正函数，确保参数在范围内(由于遗传算法在mate、mutate之后会溢出范围，需要修正)
    def repair_bounds(individual, param_ranges):
        length = len(param_ranges)
        for i, (min_val, max_val) in enumerate(param_ranges):
            individual[i] = np.clip(individual[i], min_val, max_val)
            if i == length - 1:  # this parameter is poly kernel degree, it must be an integer
                individual[i] = int(individual[i])
        return individual

    # 包装交叉操作，修正交叉后的参数范围
    def constrained_cxBlend(ind1, ind2, alpha=0.5):
        # 混合交叉，可以直接看源码，就几行。就是交叉后offspring[i]=(1. - gamma) * ind1[i] + gamma * ind2[i],gama是基于一个随机数产生
        # alpha=0.5，gama \in [-1,0.5),所以交叉后的值不会偏离父值太远
        tools.cxBlend(ind1, ind2, alpha)
        param_ranges = [C_range, epsilon_range, gamma_range, coef0_range, degree_range]
        repair_bounds(ind1, param_ranges)
        repair_bounds(ind2, param_ranges)
        return ind1, ind2

    # 包装变异操作，修正变异后的参数范围
    def constrained_mutate(individual, mu=0, sigma=0.1, indpb=0.2):
        tools.mutGaussian(individual, mu, sigma, indpb)  # 高斯变异，变异离原来的值不会差太远
        param_ranges = [C_range, epsilon_range, gamma_range, coef0_range, degree_range]
        repair_bounds(individual, param_ranges)
        return individual,

    # 创建 DEAP 遗传算法工具箱
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # 最大化目标
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    # 定义个体及其属性
    toolbox.register("attr_C", np.random.uniform, *C_range)  # 调用np.random.uniform(C_range)初始化attr_C
    toolbox.register("attr_epsilon", np.random.uniform, *epsilon_range)
    toolbox.register("attr_gamma", np.random.uniform, *gamma_range)
    toolbox.register("attr_coef0", np.random.uniform, *coef0_range)
    toolbox.register("attr_degree", np.random.randint, *degree_range)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_C, toolbox.attr_epsilon, toolbox.attr_gamma, toolbox.attr_coef0, toolbox.attr_degree), n=1)  # n=1表示生成一次(toolbox.attr_C, toolbox.attr_epsilon, toolbox.attr_gamma)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 注册遗传算法操作
    toolbox.register("select", tools.selTournament,
                     tournsize=3)  # 竞标赛选择, 在eaSimple中默认k=len(population)，即选出len(population)个用于交配
    toolbox.register("mate", constrained_cxBlend)  # 混合交叉 eaSimple中操作就是让每相邻两个parent进行交配（概率行为）
    toolbox.register("mutate", constrained_mutate)  # 高斯变异 eaSimple也是遍历所有offspring，让其变异（概率行为）
    toolbox.register("evaluate", fitness)

    # 设置 GA 参数
    population = toolbox.population(n=50)  # 种群大小
    n_generations = 20  # 迭代次数

    # 运行遗传算法 cxpb: 交叉(mate/crossover)概率。mutpb: 变异概率。环境选择采用全替换策略
    # 输出中，nevals小于population是由于mutate和mate是概率行为，有的individual没有经过mate和mutate，就不会评估
    result_population, _ = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=n_generations,
                                               verbose=True)

    # 找到最优个体
    best_individual = tools.selBest(result_population, k=1)[0]
    best_C, best_epsilon, best_gamma ,best_coef0, best_degree = best_individual
    print(
        f"最佳参数：C={best_C}, epsilon={best_epsilon}, gamma={best_gamma}, coef0={best_coef0}, degree={best_degree}")

    # 使用最佳参数训练最终模型
    final_model = SVR(kernel='poly', C=best_C, epsilon=best_epsilon, gamma=best_gamma, coef0=best_coef0, degree=best_degree)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    # 输出最终模型的性能
    mse = mean_squared_error(y_test, y_pred)
    print(f"测试集均方误差 (MSE): {mse}")
    #最佳参数：C=60.237715599516626, epsilon=0.03286928348960132, gamma=0.0005105550212938908, coef0=2.4593329741761703, degree=3

"""
根据获取的RBF核和Poly核的最优参数，获取最佳a
"""
def getBestAByGA():
    dataset = YahooDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()
    best_rbf_gamma = 0.0001
    best_poly_gamma, best_poly_coef0, best_poly_degree = 0.0005, 2.46, 3

    # 自定义混合kernel
    def mixed_kernel(X, Y, a=0.5, b=0.5):
        # RBF 核
        rbf = rbf_kernel(X, Y, gamma=best_rbf_gamma)
        # 多项式核
        poly = polynomial_kernel(X, Y, gamma=best_poly_gamma, coef0=best_poly_coef0, degree=best_poly_degree)
        # 混合核
        return a * rbf + b * poly

    # 定义遗传算法的优化目标函数
    def fitness(individual):
        a, C, epsilon = individual
        # lambda 为匿名函数，等于创造一个函数func(X,Y)，不过用匿名函数可以在程序运行时动态调整传入参数的大小 e.g.a，b
        model = SVR(kernel=lambda X, Y: mixed_kernel(X, Y, a=a, b=1 - a),
                    C=C,
                    epsilon=epsilon
                    )
        # 使用交叉验证评估模型性能(太慢了)
        # scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
        # return np.mean(scores),  # 目标是最大化负的 MSE (最小化 MSE)

        # 快一些，就fit一次
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return -mse,

    # 定义 GA 的搜索范围
    a_range = (0, 1)  # b=1-a
    C_range = (0.1, 100)  # C 的范围
    epsilon_range = (0.01, 1)  # epsilon 的范围

    # 修正函数，确保参数在范围内(由于遗传算法在mate、mutate之后会溢出范围，需要修正)
    def repair_bounds(individual, param_ranges):
        length = len(param_ranges)
        for i, (min_val, max_val) in enumerate(param_ranges):
            individual[i] = np.clip(individual[i], min_val, max_val)
        return individual

    # 包装交叉操作，修正交叉后的参数范围
    def constrained_cxBlend(ind1, ind2, alpha=0.5):
        # 混合交叉，可以直接看源码，就几行。就是交叉后offspring[i]=(1. - gamma) * ind1[i] + gamma * ind2[i],gama是基于一个随机数产生
        # alpha=0.5，gama \in [-1,0.5),所以交叉后的值不会偏离父值太远
        tools.cxBlend(ind1, ind2, alpha)
        param_ranges = [a_range, C_range, epsilon_range]
        repair_bounds(ind1, param_ranges)
        repair_bounds(ind2, param_ranges)
        return ind1, ind2

    # 包装变异操作，修正变异后的参数范围
    def constrained_mutate(individual, mu=0, sigma=0.1, indpb=0.2):
        tools.mutGaussian(individual, mu, sigma, indpb)  # 高斯变异，变异离原来的值不会差太远
        param_ranges = [a_range, C_range, epsilon_range]
        repair_bounds(individual, param_ranges)
        return individual,

    # 创建 DEAP 遗传算法工具箱
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # 最大化目标
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    # 定义个体及其属性
    toolbox.register("attr_a", np.random.uniform, *a_range)
    toolbox.register("attr_C", np.random.uniform, *C_range)  # 调用np.random.uniform(C_range)初始化attr_C
    toolbox.register("attr_epsilon", np.random.uniform, *epsilon_range)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_a, toolbox.attr_C, toolbox.attr_epsilon), n=1)  # n=1表示生成一次(toolbox.attr_C, toolbox.attr_epsilon, toolbox.attr_gamma)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 注册遗传算法操作
    toolbox.register("select", tools.selTournament,
                     tournsize=3)  # 竞标赛选择, 在eaSimple中默认k=len(population)，即选出len(population)个用于交配
    toolbox.register("mate", constrained_cxBlend)  # 混合交叉 eaSimple中操作就是让每相邻两个parent进行交配（概率行为）
    toolbox.register("mutate", constrained_mutate)  # 高斯变异 eaSimple也是遍历所有offspring，让其变异（概率行为）
    toolbox.register("evaluate", fitness)

    # 设置 GA 参数
    population = toolbox.population(n=50)  # 种群大小
    n_generations = 20  # 迭代次数

    # 运行遗传算法 cxpb: 交叉(mate/crossover)概率。mutpb: 变异概率。环境选择采用全替换策略
    # 输出中，nevals小于population是由于mutate和mate是概率行为，有的individual没有经过mate和mutate，就不会评估
    result_population, _ = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=n_generations,
                                               verbose=True)

    # 找到最优个体
    best_individual = tools.selBest(result_population, k=1)[0]
    best_a, best_C, best_epsilon = best_individual
    print(
        f"最佳参数：a={best_a} ,C={best_C}, epsilon={best_epsilon}")

    # 使用最佳参数训练最终模型
    final_model = SVR(kernel=lambda X, Y: mixed_kernel(X, Y, a=best_a, b=1 - best_a),
                      C=best_C,
                      epsilon=best_epsilon
                      )
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    # 输出最终模型的性能
    mse = mean_squared_error(y_test, y_pred)
    print(f"测试集均方误差 (MSE): {mse}")
    #最佳参数：a=0.2741834489340511 ,C=90.03709910307612, epsilon=0.03426895707483308

if __name__ == "__main__":
    Best_SVR_model()
    #GA()
    #getBestRBFParametersByGA()
    #getBestPolyParametersByGA()
    #getBestAByGA()