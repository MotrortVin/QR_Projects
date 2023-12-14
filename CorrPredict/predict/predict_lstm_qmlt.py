from params.init import *
from tools.data_load import *
from tools.model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from tqdm.notebook import tqdm, trange


def predict_corr_lstm(stock_l, start, end, i_max, adj, n, rate, features, lb, hidden_size, y_size, lr, epoch_num):
    stock_dict = {}

    # Data Preparation
    for stock in stock_l:
        # data load
        df = get_stock_price(stock, if_max=i_max, start=start, end=end)
        # df = normalization(df.loc[:, ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])
        # null value check
        assert check_na_value_by_df(df, stock)
        # return calculate
        stock_dict[stock] = get_return_by_df(df, adj=adj)

    stock_dict = corr_cal(stock_dict, n)

    # Data Preprocessing
    train_X, test_X, train_y, test_y = get_data_Xy(stock_dict, rate, features)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_X = torch.tensor(scaler.fit_transform(train_X.values))
    test_X = torch.tensor(scaler.fit_transform(test_X.values))
    train_y = torch.tensor(scaler.fit_transform(train_y.values))
    test_y = torch.tensor(scaler.fit_transform(test_y.values))

    train_X = train_X.clone().detach().to(torch.float32)
    test_X = test_X.clone().detach().to(torch.float32)
    train_y = train_y.clone().detach().to(torch.float32)
    test_y = test_y.clone().detach().to(torch.float32)
    print(train_X.dtype)

    train_sequences, train_labels = create_sequences_mimo(train_X, train_y, look_back, features)
    test_sequences, test_labels = create_sequences_mimo(test_X, test_y, look_back, features)

    model = CorrelationPredictor(len(features), hidden_size, y_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total = (len(train_labels))
    for epoch in tqdm(range(epoch_num), desc='Training Progress'):
        count = 0
        for seq, label in zip(train_sequences, train_labels):
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model.forward(seq)

            loss = criterion(y_pred, label)
            loss.backward()
            optimizer.step()
            print('\rSequences Training: {:.2%}'.format(count / total), end='')
            count += 1

    model.eval()
    with torch.no_grad():
        test_predictions = torch.zeros_like(test_labels)
        for i, seq in enumerate(test_sequences):
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            test_predictions[i] = model(seq)

    predicted_correlations = scaler.inverse_transform(test_predictions.numpy().reshape(5, -1).T)
    actual_correlations = scaler.inverse_transform(test_labels.numpy().reshape(5, -1).T)
    predicted_correlations_df = pd.DataFrame(predicted_correlations,
                                             columns=[i + ' Predicted_Correlation' for i in stock_list[:-1]])
    actual_correlations_df = pd.DataFrame(actual_correlations,
                                          columns=[i + ' Actual_Correlation' for i in stock_list[:-1]])

    result_df = pd.concat([actual_correlations_df, predicted_correlations_df], axis=1)
    print(result_df)

    sns.heatmap(result_df.corr(), cmap='viridis', annot=True, fmt=".3f")

    for stock in stock_list[:-1]:
        print(f"【{stock}】Mean Squared Error (MSE):",
              ((result_df[stock + ' Predicted_Correlation'] - result_df[stock + ' Actual_Correlation']) ** 2).mean())


if __name__ == '__main__':
    # start_date = '2005-01-01'
    # end_date = '2023-12-12'
    # if_max = False
    # actions = True
    # stock_list = ['GOOG', 'AMZN', 'JPM', 'GME', 'XOM', 'SPY']
    # period = 40
    # train_rate = 0.8
    # features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Return']
    # look_back = 80
    # input_size = len(features)
    # hidden_layer_size = 128
    # output_size = 1
    # epochs = 100
    # learning_rate = 0.001
    predict_corr_lstm(stock_list, start_date, end_date, if_max, actions, period, train_rate, feature_list, look_back, hidden_layer_size, output_size, learning_rate, epochs)