from params.init import *
from tools.data_load import *
from tools.model import *
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import yfinance as yf

# 获取股票数据
symbols = ['GOOG', 'AMZN', 'JPM', 'GME', 'XOM']
data = yf.download(symbols, start='2022-01-01', end='2023-01-01')

# 计算每日收益率
data['GOOG_Return'] = data['GOOG']['Close'].pct_change()
data['AMZN_Return'] = data['AMZN']['Close'].pct_change()
data['JPM_Return'] = data['JPM']['Close'].pct_change()
data['GME_Return'] = data['GME']['Close'].pct_change()
data['XOM_Return'] = data['XOM']['Close'].pct_change()

# 删除 NaN 值
data = data.dropna()

# 特征选择
features = ['GOOG_Open', 'GOOG_Close', 'GOOG_High', 'GOOG_Low',
            'AMZN_Open', 'AMZN_Close', 'AMZN_High', 'AMZN_Low',
            'JPM_Open', 'JPM_Close', 'JPM_High', 'JPM_Low',
            'GME_Open', 'GME_Close', 'GME_High', 'GME_Low',
            'XOM_Open', 'XOM_Close', 'XOM_High', 'XOM_Low']

# 构建特征矩阵和目标变量
X = data[features]
y = data['GOOG_Return']  # 以 GOOG 的收益率为例

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义 LightGBM 模型
params = {'objective': 'regression',
          'metric': 'mse',
          'num_leaves': 31,
          'learning_rate': 0.05,
          'feature_fraction': 0.9}

num_round = 100

# 转换数据为 LightGBM 数据集格式
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 训练模型
model = lgb.train(params, train_data, num_round, valid_sets=[train_data, test_data], early_stopping_rounds=10)

# 预测测试集
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')