start_date = '2015-01-01'
end_date = '2023-12-12'
if_max = False
actions = True
stock_list = ['GOOG', 'AMZN', 'JPM', 'GME', 'XOM', 'SPY']
feature_list = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Return']
period = 40
train_rate = 0.8

params = {'objective': 'regression',
          'metric': 'mse',
          'num_leaves': 31,
          'learning_rate': 0.05,
          'feature_fraction': 0.9}