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


def solo_data_predict(stock_df, rate, features):
    input_size = len(features)
    train_X, test_X, train_y, test_y= get_data_Xy_solo(stock_df, rate, features)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_X = torch.tensor(scaler.fit_transform(train_X.values))
    test_X = torch.tensor(scaler.fit_transform(test_X.values))
    train_y = torch.tensor(scaler.fit_transform(train_y.values))
    test_y = torch.tensor(scaler.fit_transform(test_y.values))

    train_X = train_X.clone().detach().to(torch.float32)
    test_X = test_X.clone().detach().to(torch.float32)
    train_y = train_y.clone().detach().to(torch.float32)
    test_y = test_y.clone().detach().to(torch.float32)

    train_sequences, train_labels = create_sequences(train_X, train_y, look_back)
    test_sequences, test_labels = create_sequences(test_X, test_y, look_back)

    model = CorrelationPredictor(input_size, hidden_layer_size, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total = (len(train_labels))
    for epoch in trange(epochs, desc='Training Progress'):
        count = 0
        for seq, label in zip(train_sequences, train_labels):
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model.forward(seq).reshape(-1, 1)

            loss = criterion(y_pred, label.reshape(-1, 1))
            loss.backward()
            optimizer.step()
            print('\rSequences Training: {:.2%} {}'.format(count/total, loss), end='')
            count += 1

    model.eval()
    with torch.no_grad():
      test_predictions = torch.zeros_like(test_labels)
      for i, seq in enumerate(test_sequences):
          model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                              torch.zeros(1, 1, model.hidden_layer_size))
          test_predictions[i] = model(seq)

    predicted_correlations = scaler.inverse_transform(test_predictions.numpy())
    actual_correlations = scaler.inverse_transform(test_labels.numpy())
    return predicted_correlations, actual_correlations

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

    predicted_correlations = {}
    actual_correlations = {}
    for stock in stock_list[:-1]:
        predicted_correlations[stock], actual_correlations[stock] = solo_data_predict(stock_dict[stock], train_rate, features)

    predicted_correlations_df = pd.DataFrame(columns=[i + ' Predicted_Correlation' for i in stock_list[:-1]])
    actual_correlations_df = pd.DataFrame(columns=[i + ' Actual_Correlation' for i in stock_list[:-1]])
    for stock in stock_list[:-1]:
        predicted_correlations_df[stock + ' Predicted_Correlation'] = predicted_correlations[stock].reshape(-1)
        actual_correlations_df[stock + ' Actual_Correlation'] = actual_correlations[stock].reshape(-1)

    result_df = pd.concat([actual_correlations_df, predicted_correlations_df], axis=1)


if __name__ == '__main__':
    predict_corr_lstm(stock_list, start_date, end_date, if_max, actions, period, train_rate, feature_list, look_back, hidden_layer_size, output_size, learning_rate, epochs)