import yfinance as yf
import pandas as pd
import numpy as np
import torch


# load stock price data
def get_stock_price(stock, if_max=False, start='2022-01-01', end='2023-01-01'):
    if if_max:
        return yf.download(stock, period='max', actions=True)
    else:
        return yf.download(stock, start=start, end=end, actions=True)


# normalization
def normalization(stock_df):
    for column in stock_df.columns:
        stock_df['column'] = (stock_df['column'] - stock_df['column'].mean()) / stock_df['column'].std()
    return stock_df


# check if there is any NA value
def check_na_value(stock_dict):
    for stock in stock_dict.keys():
        if len(stock_dict[stock]) != len(stock_dict[stock].dropna()):
            print(stock, 'null!')
            return False
    print('NULL value Checked!')
    return True


def check_na_value_by_df(stock_df, stock):
    if len(stock_df) != len(stock_df.dropna()):
        print(stock, 'null!')
        return False
    print('NULL value Checked!')
    return True


# get return
def get_return(stock_dict, adj=False):
    label = 'Adj Close' if adj else 'Close'
    for stock in stock_dict.keys():
        stock_dict[stock]['Return'] = stock_dict[stock][label].pct_change().fillna(0)
    return stock_dict


def get_return_by_df(stock_df, adj=False):
    label = 'Adj Close' if adj else 'Close'
    stock_df['Return'] = stock_df[label].pct_change().fillna(0)
    return stock_df


# get corr
def corr_cal(stock_dict, n):
    for stock in stock_dict.keys():
        stock_dict[stock]['Corr_p'] = stock_dict[stock]['Return'].rolling(n).corr(
            stock_dict['SPY']['Return']).shift(-n + 1)
        stock_dict[stock] = stock_dict[stock].iloc[:-n + 1, :]
    return stock_dict


# get X and y
def get_data_Xy(stock_dict, rate, features):
    train_size = int(len(stock_dict['SPY']) * rate)
    data_X = pd.concat(
        [stock_dict[i][features] for i in stock_dict.keys() if i != 'SPY'],
        axis=1)
    data_y = pd.concat([stock_dict[i]['Corr_p'] for i in stock_dict.keys() if i != 'SPY'], axis=1)
    # data_X = pd.concat(
    #     [stock_dict[i][features] for i in stock_dict.keys()],
    #     axis=1)
    # data_y = pd.concat([stock_dict[i]['Corr_p'] for i in stock_dict.keys()], axis=1)
    return data_X.iloc[:train_size, :], data_X.iloc[train_size:, :], data_y.iloc[:train_size, :], data_y.iloc[
                                                                                                  train_size:, :]


def create_sequences(data, target, look_back):
    sequences, labels = [], []
    for i in range(len(data) - look_back):
        sequence = data[i:i + look_back]
        label = target[i + look_back]
        sequences.append(sequence)
        labels.append(label)
    return torch.stack(sequences), torch.stack(labels)


def create_sequences_mimo(data, target, look_back, features):
    sequences, labels = [], []
    fn = int(np.shape(data)[1] / features)
    for k in range(fn):
        for i in range(len(data) - look_back):
            sequence = data[i:i + look_back][(k * fn): (k * fn + fn)]
            label = target[i + look_back][k]
            sequences.append(sequence)
            labels.append(label)
    return torch.stack(sequences), torch.stack(labels)


def get_data_Xy_solo(stock_df, rate, features):
    train_size = int(len(stock_df) * rate)
    data_X = stock_df[features]
    data_y = stock_df[['Corr_p']]
    return data_X.iloc[:train_size, :], data_X.iloc[train_size:, :], data_y.iloc[:train_size, :], data_y.iloc[train_size:, :]

