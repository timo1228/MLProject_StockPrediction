"""
author: Yiting Chen
description: using SVR to implement stock prediction, using Genetic Algorithm to select best parameters of SVR
"""

from DataSource import YahooDataSet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score

#SVR
from sklearn.svm import SVR
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

#deap
from deap import base, creator, tools, algorithms

def SVR_model():
    dataset = YahooDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()

    def mixed_kernel(X, Y, a=0.5, b=0.5, gamma=0.1, coef0=1, degree=10):
        # RBF 核
        rbf = rbf_kernel(X, Y, gamma=gamma)
        # 多项式核
        poly = polynomial_kernel(X, Y, gamma=gamma, coef0=coef0, degree=degree)
        # 混合核
        return a * rbf + b * poly

    # 初始化SVR模型，epsilon是惩罚平面的距离，超过这个距离就会被penalize
    #k(x, x')= exp(-gamma ||x - x'||^2), 较小的 gamma 值：核函数的范围更大（对距离更不敏感）。每个数据点的影响范围更宽广。模型更倾向于低复杂度，可能会欠拟合。
    #svr = SVR(kernel='rbf', C=100, gamma=0.0001, epsilon=1)
    #lambda 为匿名函数，等于创造一个函数func(X,Y)，不过用匿名函数可以在程序运行时动态调整传入参数的大小 e.g.a，b
    #svr = SVR(kernel=lambda X, Y: mixed_kernel(X, Y, a=0.5, b=0.5, gamma=0.0003, coef0=1, degree=3),
    #          C=100,
    #          epsilon=0.01
    #          )
    """
    so slow parameters
    calculate fitness：a=0.5446087278524019 ,C=24.70641448944032, epsilon=0.20258273857779255, gamma=0.8542544571307036, coef0=1.355690711748752, degree=4
    """
    #最佳参数：a=0.0 ,C=82.19160237270314, epsilon=0.12925426387806566, gamma=0.8374411730996684, coef0=1.6091375413790112, degree=1
    svr = SVR(kernel=lambda X, Y: mixed_kernel(X, Y, a=0, b=1, gamma=0.8374411730996684, coef0=1.6091375413790112, degree=1),
              C=82.19160237270314,
              epsilon=0.12925426387806566
              )

    # 训练模型
    svr.fit(X_train, y_train.ravel())

    # 预测
    y_pred_test = svr.predict(X_test)
    y_pred_train = svr.predict(X_train)
    print(f'mean_squared_error on test set = {mean_squared_error(y_test, y_pred_test)}')

    df_data = dataset.data
    y_origin = np.concatenate([y_train,  y_test])
    y_pred_show = np.concatenate([y_pred_train, y_pred_test])
    # 创建折线图
    #plt.plot(df_data.index, y_origin, label='Line', color='b', marker='o',linewidth=0.01)  # label 是图例，color 是线条颜色，marker 是数据点的标记
    #plt.plot(df_data.index, y_pred_show, label='Line', color='r', marker='o', linewidth=0.01)

    n_rows = len(df_data)
    plt.plot(df_data.index[int(0.8 * n_rows):], y_pred_test, label='prediction', color='b', marker='o', linewidth=0.01)  # label 是图例，color 是线条颜色，marker 是数据点的标记
    plt.plot(df_data.index[int(0.8 * n_rows):], y_test, label='truth', color='r', marker='o', linewidth=0.01)



    # 添加标题和标签
    plt.title('SVR Plot')  # 图表标题
    plt.xlabel('X Axis')  # X轴标签
    plt.ylabel('Y Axis')  # Y轴标签

    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()

def GA():
    dataset = YahooDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()

    #自定义混合kernel
    def mixed_kernel(X, Y, a=0.5, b=0.5, gamma=0.1, coef0=1, degree=3):
        # RBF 核
        rbf = rbf_kernel(X, Y, gamma=gamma)
        # 多项式核
        poly = polynomial_kernel(X, Y, gamma=gamma, coef0=coef0, degree=degree)
        # 混合核
        return a * rbf + b * poly

    # 定义遗传算法的优化目标函数
    def fitness(individual):
        a, C, epsilon, gamma, coef0, degree = individual
        #print(f"calculate fitness：a={a} ,C={C}, epsilon={epsilon}, gamma={gamma}, coef0={coef0}, degree={degree}")
        #model = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)
        # lambda 为匿名函数，等于创造一个函数func(X,Y)，不过用匿名函数可以在程序运行时动态调整传入参数的大小 e.g.a，b
        model = SVR(kernel=lambda X, Y: mixed_kernel(X, Y, a=a, b=1-a, gamma=gamma, coef0=coef0, degree=degree),
                  C=C,
                  epsilon=epsilon
                  )
        # 使用交叉验证评估模型性能
        #scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return -mse,
        #return np.mean(scores),  # 目标是最大化负的 MSE (最小化 MSE)

    # 定义 GA 的搜索范围
    a_range = (0,1) #b=1-a

    C_range = (0.1, 100)  # C 的范围
    epsilon_range = (0.01, 1)  # epsilon 的范围
    #rbf kernel超参数
    gamma_range = (0.00001, 1)  # gamma 的范围
    #polu kernel超参数
    coef0_range = (-1, 3)
    degree_range = (1, 5)

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
        param_ranges = [a_range , C_range, epsilon_range, gamma_range, coef0_range, degree_range]
        repair_bounds(ind1, param_ranges)
        repair_bounds(ind2, param_ranges)
        return ind1, ind2

    # 包装变异操作，修正变异后的参数范围
    def constrained_mutate(individual, mu=0, sigma=0.1, indpb=0.2):
        tools.mutGaussian(individual, mu, sigma, indpb) #高斯变异，变异离原来的值不会差太远
        param_ranges = [a_range ,C_range, epsilon_range, gamma_range, coef0_range, degree_range]
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
    toolbox.register("attr_gamma", np.random.uniform, *gamma_range)
    toolbox.register("attr_coef0", np.random.uniform, *coef0_range)
    toolbox.register("attr_degree", np.random.randint, *degree_range)


    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_a, toolbox.attr_C, toolbox.attr_epsilon, toolbox.attr_gamma, toolbox.attr_coef0, toolbox.attr_degree), n=1) #n=1表示生成一次(toolbox.attr_C, toolbox.attr_epsilon, toolbox.attr_gamma)
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
    best_a, best_C, best_epsilon, best_gamma, best_coef0, best_degree = best_individual
    print(f"最佳参数：a={best_a} ,C={best_C}, epsilon={best_epsilon}, gamma={best_gamma}, coef0={best_coef0}, degree={best_degree}")

    # 使用最佳参数训练最终模型
    #final_model = SVR(kernel='rbf', C=best_C, epsilon=best_epsilon, gamma=best_gamma)
    final_model = SVR(kernel=lambda X, Y: mixed_kernel(X, Y, a=best_a, b=1 - best_a, gamma=best_gamma, coef0=best_coef0, degree=best_degree),
                C=best_C,
                epsilon=best_epsilon
                )
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    # 输出最终模型的性能
    mse = mean_squared_error(y_test, y_pred)
    print(f"测试集均方误差 (MSE): {mse}")

if __name__ == "__main__":
    SVR_model()
    #GA()