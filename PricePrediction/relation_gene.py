import pywt
import numpy as np
import pandas as pd
import seaborn as sns
from configs import *
import scipy.fft as fft
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def industry_corr():
    industry = pd.read_csv(raw_data_path + 'industry_code_all.csv', index_col=0)
    close_hfq = pd.read_csv(raw_data_path + 'hfq_close.csv', index_col=0)
    industry['date'] = industry.index
    industry['date'] = pd.to_datetime(industry['date'])
    industry = industry.set_index('date')
    close_hfq['date'] = close_hfq.index
    close_hfq['date'] = pd.to_datetime(close_hfq['date'])
    close_hfq = close_hfq.set_index('date')

    for i in [2017, 2018, 2019, 2020, 2021]:
        industry_cut = industry.loc[str(i):str(i), :]
        close_hfq_cut = close_hfq.loc[str(i):str(i), :]
        first_layer = industry_cut.iloc[-1:, :]
        first_layer['title'] = 'industry'
        first_layer = first_layer.set_index('title')
        first_layer = first_layer.T
        first_layer = first_layer.dropna(axis=0, how='any')
        first_layer['code'] = first_layer.index
        for industry_name, industry_group in first_layer.groupby('industry'):
            industry_list = list(industry_group['code'])
            industry_close = close_hfq_cut[industry_list]
            corr_matrix = industry_close.corr()
            corr_matrix.to_csv(output_path + f'corr/{i}_{industry_name}_corr.csv')
            if len(industry_close.columns) < 30:
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
                plt.savefig(output_path + f'corr/{i}_{industry_name}_corr.png')


def vector_clustering(vectors, num_clusters):
    # 创建 K-means 模型
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(vectors)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    return cluster_labels, cluster_centers


def wave_class():
    close_hfq = pd.read_csv(raw_data_path + 'hfq_close.csv', index_col=0)
    close_hfq['date'] = close_hfq.index
    close_hfq['date'] = pd.to_datetime(close_hfq['date'])
    wavelet = 'db4'  # 选择小波函数
    level = 5  # 分解的层数

    for year, close_hfq_cut in close_hfq.groupby(pd.Grouper(key='date', freq='y')):
        close_hfq_cut = close_hfq_cut.dropna(axis=1, how='any')
        param_df = pd.DataFrame(columns=list(range(28)))
        for code in close_hfq_cut.columns:
            try:
                # 计算收益率
                return_cut = 1 - close_hfq_cut[code] / close_hfq_cut[code].shift(1)
                # 平滑处理
                return_cut = return_cut.fillna(method='ffill').fillna(method='bfill')
                # 进行小波变换
                coeffs = pywt.wavedec(np.array(return_cut), wavelet, level=level)
                params = np.concatenate((coeffs[0], coeffs[1]), axis=0)

                param_df.loc[code] = params
            except Exception as e:
                print(code)

        num_clusters = 4
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(param_df)

        param_df['label'] = kmeans.labels_
        param_df.to_csv(output_path + f'wave/{str(year)[:4]}_kmeans_4_labels.csv')


def volatility_class():
    close_hfq = pd.read_csv(raw_data_path + 'hfq_close.csv', index_col=0)
    close_hfq['date'] = close_hfq.index
    close_hfq['date'] = pd.to_datetime(close_hfq['date'])
    close_hfq = close_hfq.set_index('date')
    vola_df = pd.DataFrame(columns=list(close_hfq.columns))
    for code in close_hfq.columns:
        return_cut = 1 - close_hfq[code] / close_hfq[code].shift(1)
        vola = return_cut.rolling(20).std()
        vola_df[code] = vola.shift(1)
    vola_df.to_csv(output_path + f'vol/rolling_volatility.csv')


if __name__ == '__main__':
    # wave_class()
    # industry_corr()
    volatility_class()