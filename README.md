1.
我们调用数据集的时候统一用DataSource写的接口，要实现不同的DataSet需要继承DataSet抽象类
```python
from DataSource import YahooDataSet

dataset = YahooDataSet()
X_train, X_test, y_train, y_test = dataset.train_and_test()
```
数据增加了"Daily_Return","'Volatility","Rolling_Mean_Close"三个feature  
对每个feature进行了标准化处理  
训练集和测试集划分是按照时间顺序8:2划分，比如10天就前8天为训练集，后2天为测试集