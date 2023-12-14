from configs import *
import numpy as np
import pandas as pd
from data_source import get_stock_list


def ts_corr(df_1, df_2, length):
    return df_1.rolling(length).corr(df_2)


def ts_vola(df, length):
    return df.rolling(length).std() / df.rolling(length).mean()


def ts_delta(df, length):
    return df - df.shift(length)


def ts_max(df, length):
    return df.rolling(length).max()


def ts_min(df, length):
    return df.rolling(length).min()


def ts_mean(df, length):
    return df.rolling(length).mean()


def ts_std(df, length):
    return df.rolling(length).std()


def ts_rank(df, length):
    return df.rolling(length).apply(lambda x: pd.Series(x).rank(ascending=False).values[-1])


def c_v_corr_fct(df): #
    return -ts_corr(df.close_hfq, df.vol, 30).shift(1)  # 避免误用未来数据


def a_s_div_fct(df):
    a_s_div = (df.total_assets / df.operate_profit) / ts_vola(df.vol, 30).shift(1)  # 避免误用未来数据
    a_s_div.replace([np.inf, -np.inf], np.nan, inplace=True)
    return a_s_div


def h_l_delta_fct(df):
    return -ts_delta(
        (((df.close_hfq - df.low_hfq) - (df.high_hfq - df.close_hfq)) / (df.high_hfq - df.low_hfq)).shift(1), 1)


def h_v_tsmax_fct(df):
    return -1 * ts_rank(ts_corr(ts_rank(df.vol, 5), ts_rank(df.high_hfq, 5), 5), 3).shift(1)


def v_p_sqrt_fct(df):
    return ((df.high * df.low).apply(np.sqrt) - 10 * df.amount / df.vol).shift(1)


def c_l_delay_fct(df):
    return ts_delta(df.close_hfq, 5).shift(1)


def k_u_ret_fct(df):
    return (df.open_hfq / df.close_hfq.shift(1) - 1).shift(1)


def l_l_chk_fct(df):
    fct = pd.concat([(ts_delta(df.close_hfq, 5) / df.close_hfq.shift(5))[df.close_hfq < df.close_hfq.shift(5)],
                     (df.close_hfq * 0)[df.close_hfq == df.close_hfq.shift(5)],
                     (ts_delta(df.close_hfq, 5) / df.close_hfq)[
                         df.close_hfq > df.close_hfq.shift(5)]], axis=0)
    fct = fct.sort_index().shift(1)
    return fct


def w_p_max_fct(df):
    return -ts_max(ts_delta(10 * df.amount / df.vol, 3), 5).shift(1)


def m_t_std_fct(df):
    return ts_std(df.amount, 6).shift(1)
