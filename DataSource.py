import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class DataSet(object):
    def __init__(self):
        path = "./data/nasdq.csv"
        self.inited = False
        #init the data
        data = pd.read_csv(path)

        # Date column: Convert to datetime format
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)

        # Rename columns to remove spaces
        data.columns = data.columns.str.replace(' ', '')

        # Feature Engineer
        # Add features such as daily returns and rolling averages
        data['Daily_Return'] = data['Close'].pct_change()  # pct_change() 计算 Close 列中每天价格的百分比变化，即每日收益率。
        # rolling(window=30) 对数据进行滚动窗口操作，这里窗口大小为30，表示过去30天的数据。 std() 计算每个30天窗口内 Close 价格的标准差，即波动率。
        data['Volatility'] = data['Close'].rolling(window=30).std()
        # rolling(window=30) 进行滚动窗口操作。 mean() 计算每个30天窗口内 Close 价格的均值。
        data['Rolling_Mean_Close'] = data['Close'].rolling(window=30).mean()
        data.dropna(inplace=True)

        self.data = data

        # Train-Test Split and Scaling
        # Define target variable and features
        X = data.drop(['Close'], axis=1)  # Ensure 'Close' is dropped to create the feature set
        y = data['Close']  # Target variable is 'Close' price

        # Step 1: Replace infinite values with NaN
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Step 2: Check for NaN values and handle them
        # Using forward fill to handle NaN values (you can adjust this as needed)
        X.fillna(method='ffill', inplace=True)

        self.X = X
        self.y = y

        # Step 3: Split the data into training and testing sets
        train_size = int(len(data) * 0.8)  # 80% 用于训练
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  #random split the data, not appropriate for time series data

        # Step 4: Standardize the data
        # 标准化数据：x'=(x-u)/sigma, u is the mean, sigma is the standard deviation
        scaler = StandardScaler()
        # fit_transform() 方法用于在训练集 X_train 上计算标准化所需的均值和标准差，并使用这些统计量对 X_train 进行标准化。
        # fit() 计算训练集的均值和标准差。transform() 根据计算的均值和标准差将训练数据转换为标准化后的数据。
        X_train_scaled = scaler.fit_transform(X_train)
        # transform() 方法用于使用在 X_train 上计算的均值和标准差来标准化测试集 X_test。注意，这里没有调用 fit()，因为测试集的标准化应该基于训练集的统计信息，而不是重新计算。
        X_test_scaled = scaler.transform(X_test)
        print("Data scaling successful!")

        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test

        self.inited = True
        print("initializing successfully!")


    def EDA(self): #展示数据的特征
        data = self.data
        print(data.head())

        # Check for missing values and data types
        data.info()

        # Check for null values and basic statistics
        data.describe()
        data.isnull().sum()

        # Plot Open, High, Low, Close over time
        fig, ax = plt.subplots(2, 2, figsize=(16, 12))
        data['Open'].plot(ax=ax[0, 0], title="Opening Price Over Time")
        data['High'].plot(ax=ax[0, 1], title="Highest Price Over Time")
        data['Low'].plot(ax=ax[1, 0], title="Lowest Price Over Time")
        data['Close'].plot(ax=ax[1, 1], title="Closing Price Over Time")
        plt.tight_layout()
        plt.show()

        # Correlation Matrix
        correlation_matrix = data.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix of NASDAQ Dataset")
        plt.show()

    def train_and_test(self):
        if self.inited:
            return self.X_train, self.X_test, self.y_train, self.y_test
        else:
            raise RuntimeError('Dataset not yet initialized!')


def test():
    dataset = DataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()

    # Train a baseline model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Make predictions
    y_pred_lr = lr.predict(X_test)

    # Evaluate the model
    print("Linear Regression R2 Score:", r2_score(y_test, y_pred_lr))
    print("Linear Regression RMSE:", mean_squared_error(y_test, y_pred_lr, squared=False))


if __name__ == "__main__":
    test()