from configs import *
import tushare as ts
import pandas as pd
import os
from tqdm import *


def get_stock_list():
    df = pd.read_csv(raw_data_path + 'close.csv', index_col=0)
    stock_list = []
    for label in df.columns:
        if not df[label].isnull().all():
            stock_list.append(label)
    return stock_list


def get_trade_date():
    df = pd.read_csv(raw_data_path + 'close.csv', index_col=0)
    df['date'] = df.index.astype('str')
    df['date'] = df['date'].apply(lambda x: x.replace('-', ''))
    return list(df['date'].astype('int'))


def df_deal(df, index_name, adj_date):
    df[index_name] = df[index_name].astype('int')
    df = df.drop_duplicates(subset=index_name, keep='first')
    df = df.set_index(index_name)
    df = df.sort_index()
    before_date = df.loc[:int(adj_date)]
    after_date = df.loc[int(adj_date):]
    if len(before_date) != 0:
        after_date.loc[int(adj_date)] = before_date.iloc[-1]
        after_date.sort_index()
    return after_date


def get_trade_data(stock_list, _start_date, _end_date):
    pro = ts.pro_api(tushare_token)
    if not os.path.exists(raw_data_path + 'stocks'):
        os.mkdir(raw_data_path + 'stocks')
    for item in tqdm(stock_list):
        df = pro.stk_factor(ts_code=item, start_date=_start_date, end_date=_end_date,
                            fields='ts_code,trade_date,close,open,high,low,vol,amount,'
                                   'adj_factor,close_qfq,close_hfq,open_qfq,open_hfq,'
                                   'high_qfq,high_hfq,low_qfq, low_hfq')
        df.sort_values('trade_date', inplace=True)
        df = df.set_index('trade_date')
        df.to_csv(raw_data_path + f'/stocks/{item}_trade_data.csv')


def get_capital_data(stock_list, adj_date, _start_date, _end_date):
    pro = ts.pro_api(tushare_token)
    if not os.path.exists(raw_data_path + '/overall'):
        os.mkdir(raw_data_path + '/overall')
    for item in tqdm(stock_list[230+525+2345+406:]):
        assets = pro.balancesheet_vip(ts_code=item, start_date=_start_date, end_date=_end_date,
                                  fields='ts_code,ann_date,f_ann_date,end_date,report_type,'
                                         'comp_type,total_assets')
        assets = df_deal(assets, 'f_ann_date', adj_date)
        others = pro.income_vip(ts_code=item, start_date=_start_date, end_date=_end_date,
                            fields='ts_code,ann_date,f_ann_date,end_date,report_type,'
                                   'comp_type,basic_eps,diluted_eps,ebit,operate_profit')
        others = df_deal(others, 'f_ann_date', adj_date)
        df = pd.read_csv(raw_data_path + f'stocks/{item}_trade_data.csv', index_col=0)
        df['total_assets'] = assets['total_assets']
        df['total_assets'] = df['total_assets'].fillna(method='ffill')
        df[['ebit', 'operate_profit', 'basic_eps', 'diluted_eps']] = others[['ebit', 'operate_profit', 'basic_eps', 'diluted_eps']]
        df[['ebit', 'operate_profit', 'basic_eps', 'diluted_eps']] = df[['ebit', 'operate_profit', 'basic_eps', 'diluted_eps']].fillna(method='ffill')
        df.to_csv(raw_data_path + f'/overall/{item}_all.csv')


if __name__ == '__main__':
    # get_trade_data(get_stock_list(), start_date, end_date)
    get_capital_data(get_stock_list(), '20170103', '20160101', end_date)
