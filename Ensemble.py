import numpy as np
from DataSource import YahooDataSet

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from xgboost import plot_importance, plot_tree
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error

"""
using GridSearch to find best params of XGBoost
"""
def XGBoost_BestParam_Search():
    dataset = YahooDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()


    parameters = {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.001, 0.005, 0.01, 0.05],
        'max_depth': [3, 5, 8, 10, 12, 15],
        'gamma': [0.001, 0.005, 0.01, 0.02],
        'random_state': [42]
    }

    #Regression Tree用做base learner用来boosting
    model = xgb.XGBRegressor(objective='reg:squarederror', verbose=True)
    #GridSearchCV会在我们定义的parameters这样的超参数筛选出最优参数
    clf = GridSearchCV(model, parameters)
    clf.fit(X_train, y_train)

    #Best params: {'gamma': 0.001, 'learning_rate': 0.05, 'max_depth': 10, 'n_estimators': 300, 'random_state': 42}
    #Best validation score =  0.030808857307457416

    print(f'Best params: {clf.best_params_}')
    print(f'Best validation score = {clf.best_score_}')

    model = xgb.XGBRegressor(**clf.best_params_, objective='reg:squarederror')
    model.fit(X_train, y_train)

    plot_importance(model)
    plt.show()

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    print(f'y_true = {np.array(y_test)[:5]}')
    print(f'y_pred = {y_pred_test[:5]}')
    print(f'mean_squared_error on test set = {mean_squared_error(y_test, y_pred_test)}')

    df_data = dataset.data
    y_origin = np.concatenate([y_train, y_test])
    y_pred_show = np.concatenate([y_pred_train, y_pred_test])
    # 创建折线图
    plt.plot(df_data.index, y_origin, label='Truth', color='b', marker='o',
             linewidth=0.01)  # label 是图例，color 是线条颜色，marker 是数据点的标记
    plt.plot(df_data.index, y_pred_show, label='Prediction', color='r', marker='o', linewidth=0.01)

    # 添加标题和标签
    plt.title('XGBoost Plot')  # 图表标题
    plt.xlabel('X Axis')  # X轴标签
    plt.ylabel('Y Axis')  # Y轴标签

    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()

def Best_XGBoost_model():
    dataset = YahooDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()

    #Best params: {'gamma': 0.001, 'learning_rate': 0.05, 'max_depth': 10, 'n_estimators': 300, 'random_state': 42}
    best_params = {
        'n_estimators': 300,
        'max_depth': 10,
        'learning_rate': 0.05,
        'gamma': 0.001,
        'random_state': 42
    }

    model = xgb.XGBRegressor(**best_params, objective='reg:squarederror')
    model.fit(X_train, y_train, verbose=False)

    plot_importance(model)
    plt.show()

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    print(f'y_true = {np.array(y_test)[:5]}')
    print(f'y_pred = {y_pred_test[:5]}')
    print(f'mean_squared_error on train set = {mean_squared_error(y_train, y_pred_train)}')
    print(f'mean_squared_error on test set = {mean_squared_error(y_test, y_pred_test)}')


    df_data = dataset.data
    n_rows = len(df_data)
    train_indices = df_data.index[0: int(0.8 * n_rows)]
    test_indices = df_data.index[int(0.8 * n_rows):]

    # 设置画布大小
    plt.figure(figsize=(12, 6))

    # 绘制prediction on training set
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
    plt.title('XGBoost Prediction vs Truth on Training Set', fontsize=16)  # 标题和字体大小
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
    plt.title('XGBoost Prediction vs Truth on Test Set', fontsize=16)  # 标题和字体大小
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Price', fontsize=12)

    # 设置图例
    plt.legend(fontsize=12)

    # 显示网格线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 显示图形
    plt.tight_layout()
    plt.show()

class StackingEnsemble(object):
    def __init__(self):
        self.base_learners = None #List of base learners(must be already trainer)
        self.meta_learner = None #meta learner
        self.initialized = False
        self.fitted = False

    def setLearner(self, base_learners, meta_learner):
        # 检查基础学习器是否提供predict方法
        for model in base_learners:
            if not hasattr(model, "predict"):
                raise ValueError("Base learner must implement a 'predict' method.")
        self.base_learners = base_learners
        self.meta_learner = meta_learner
        self.initialized = True
        return self

    def fit(self, X, y):
        if not self.initialized:
            raise Exception("StackingEnsemble is not initialized. Call setLearner first.")
        base_learners_preds = []
        try:
            for model in self.base_learners:
                model_pred = model.predict(X)
                base_learners_preds.append(model_pred)

            meta_X_train = np.column_stack(base_learners_preds)

            #fit meta learner
            self.meta_learner.fit(meta_X_train, y)
            self.fitted = True
            return self
        except(Exception) as e:
            print("Base learner has problems!")

    def predict(self, X):
        if not self.initialized or not self.fitted:
            raise Exception("Stacking Ensemble Model has not been fitted yet!")

        base_learners_preds = []
        try:
            for model in self.base_learners:
                model_pred = model.predict(X)
                base_learners_preds.append(model_pred)

            meta_X = np.column_stack(base_learners_preds)

            #use mata learner to pred
            y_pred = self.meta_learner.predict(meta_X)
            return y_pred
        except(Exception) as e:
            print("Base learner has problems!")

def stacking_ensemble():
    dataset = YahooDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()

    base_learners = []

    xgb_best_params = {
        'n_estimators': 300,
        'max_depth': 10,
        'learning_rate': 0.05,
        'gamma': 0.001,
        'random_state': 42
    }

    xgb_model = xgb.XGBRegressor(**xgb_best_params, objective='reg:squarederror')
    xgb_model.fit(X_train, y_train, verbose=False)
    xgb_pred = xgb_model.predict(X_test)
    xgb_mse = mean_squared_error(y_test, xgb_pred)
    print(f"xbg测试集均方误差 (MSE): {xgb_mse}")
    base_learners.append(xgb_model)

    svr_poly = SVR(kernel='poly', C=60, epsilon=0.033, gamma=0.0005, coef0=2.46, degree=3)
    svr_poly.fit(X_train, y_train)
    svr_poly_pred = svr_poly.predict(X_test)
    svr_poly_mse = mean_squared_error(y_test, svr_poly_pred)
    print(f"svr_poly测试集均方误差 (MSE): {svr_poly_mse}")
    base_learners.append(svr_poly)

    svr_rbf = SVR(kernel='rbf', C=100, epsilon=0.012, gamma=0.00015)
    svr_rbf.fit(X_train, y_train)
    svr_rbf_pred = svr_rbf.predict(X_test)
    svr_rbf_mse = mean_squared_error(y_test, svr_rbf_pred)
    print(f"svr_rbf测试集均方误差 (MSE): {svr_rbf_mse}")
    base_learners.append(svr_rbf)

    #meta_learner = LinearRegression()
    meta_learner = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
    #meta_learner = Ridge(alpha=0.1)


    stacking_ensemble = StackingEnsemble()
    stacking_ensemble.setLearner(base_learners, meta_learner)
    stacking_ensemble.fit(X_train, y_train)
    y_pred = stacking_ensemble.predict(X_test)

    # 输出最终模型的性能
    mse = mean_squared_error(y_test, y_pred)
    print(f"测试集均方误差 (MSE): {mse}")

if __name__ == "__main__":
    #stacking_ensemble()
    #XGBoost_BestParam_Search()
    Best_XGBoost_model()