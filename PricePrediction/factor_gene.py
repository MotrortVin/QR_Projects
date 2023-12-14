from factors import *
import pandas as pd
import numpy as np
from tqdm import *
import os
import alphalens
from data_source import get_stock_list, get_trade_date


class Factor:
    stock_list: list = []
    fact_name: str = ''
    factor: pd.DataFrame = None
    hfq_open: pd.DataFrame = None
    multi_index_factor: pd.DataFrame = None

    def __init__(self, fact):
        self.stock_list = get_stock_list()
        self.fact = fact
        self.fact_name = str(fact).split(' ')[1]

        try:
            self.hfq_open = pd.read_csv(raw_data_path + 'hfq_open.csv', index_col=0)
            self.hfq_open['new_index'] = self.hfq_open.index
            self.hfq_open['new_index'] = pd.to_datetime(self.hfq_open['new_index'])
            self.hfq_open = self.hfq_open.set_index('new_index')
        except Exception as e:
            print(f'【notice】hfq_open read failed.', e)
            self.hfq_open = pd.DataFrame(index=get_trade_date())

        try:
            self.factor = pd.read_csv(raw_data_path + f'factor/{self.fact_name}.csv', index_col='date')
        except Exception as e:
            print(f'【notice】factor read failed.', e)
            self.factor = pd.DataFrame(index=get_trade_date())

        try:
            self.multi_index_factor = pd.read_csv(raw_data_path + f'factor/_{self.fact_name}.csv')
            self.multi_index_factor['date'] = pd.to_datetime(self.multi_index_factor['date'])
            self.multi_index_factor = self.multi_index_factor.set_index(['date', 'asset'])
        except Exception as e:
            print(f'【notice】multi_index_factor read failed.', e)
            self.multi_index_factor = pd.DataFrame(index=get_trade_date())

    def function_gene(self, df):
        return self.fact(df)

    def iterator(self):
        if not os.path.exists(raw_data_path + f'factor/'):
            os.mkdir(raw_data_path + f'factor/')

        for label in tqdm(self.stock_list):
            fact = pd.read_csv(raw_data_path + f'overall/{label}_all.csv', index_col=0)
            self.factor[label] = self.function_gene(fact)

        self.factor = self.factor.fillna(method='ffill')
        self.factor.to_csv(raw_data_path + f'factor/{self.fact_name}.csv')

    def get_hfq_open_price(self):
        for label in tqdm(self.stock_list):
            fact = pd.read_csv(raw_data_path + f'overall/{label}_all.csv', index_col=0)
            self.hfq_open[label] = fact['open_hfq']

        self.hfq_open['new_index'] = self.hfq_open.index.astype('str')
        self.hfq_open['new_index'] = pd.to_datetime(self.hfq_open['new_index']).dt.strftime('%Y-%m-%d')
        self.hfq_open = self.hfq_open.set_index('new_index')
        self.hfq_open.index.set_names('date')
        self.hfq_open.to_csv(raw_data_path + 'hfq_open.csv')

    def df_form_trans(self):
        index = list(self.factor.index)
        for i in tqdm(range(len(self.factor))):
            single_day_fact = self.factor.iloc[i, :]
            single_day_fact = single_day_fact.T
            single_day_fact.name = str(index[i])[:4] + '-' + str(index[i])[4:6] + '-' + str(index[i])[6:]
            self.multi_index_factor = self.multi_index_factor.append(single_day_fact)

        self.multi_index_factor.index.set_names(['date'], inplace=True)
        self.multi_index_factor = self.multi_index_factor.stack()
        self.multi_index_factor = pd.DataFrame(self.multi_index_factor)
        # ## 设置索引名称
        self.multi_index_factor.index.set_names(['date', 'asset'], inplace=True)
        # ## 设置列名称
        self.multi_index_factor.columns = ['factor_value']
        self.multi_index_factor.to_csv(raw_data_path + f'factor/_{self.fact_name}.csv')

    def factor_analysis(self):
        gap = len(self.hfq_open.index) - len(self.multi_index_factor.index.levels[0])
        data = alphalens.utils.get_clean_factor_and_forward_returns(self.multi_index_factor, self.hfq_open.iloc[gap:],
                                                                    quantiles=5,
                                                                    periods=(1, 5, 10, 20))
        alphalens.tears.create_full_tear_sheet(data)


factor = Factor(m_t_std_fct)
factor.iterator()
factor.df_form_trans()
# factor_0.get_hfq_open_price()
# factor_0.factor_analysis()
