from configs import *
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

close_data = pd.read_csv(raw_data_path+"close.csv", index_col=0)
volume_data = pd.read_csv(raw_data_path+"volume.csv", index_col=0)
adj_factor_data = pd.read_csv(raw_data_path+"adjfactor.csv", index_col=0)

hfq_close_data = close_data * adj_factor_data
hfq_close_data.to_csv(raw_data_path+"hfq_close.csv")

qfq_close_data = close_data * adj_factor_data / adj_factor_data.iloc[-1]
qfq_close_data.to_csv(raw_data_path+"qfq_close.csv")