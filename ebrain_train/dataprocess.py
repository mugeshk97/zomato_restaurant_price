import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sqlite3


class Preprocess:
    def __init__(self):
        pass

    def export_data(self, database):
        con = sqlite3.connect(database)
        dataframe = pd.read_sql_query("SELECT * FROM data", con)
        return dataframe

    def cost(self, dataframe):
        dataframe['cost'] = dataframe['cost'].astype(str)
        dataframe['cost'] = dataframe['cost'].apply(lambda x: x.replace(',', ''))
        dataframe['cost'] = dataframe['cost'].astype(float)
        return dataframe

    def rating(self, dataframe):
        dataframe = dataframe.loc[dataframe.rate != 'NEW']
        dataframe = dataframe.loc[dataframe.rate != '-'].reset_index(drop=True)
        remove_slash = lambda x: x.replace('/5', '') if type(x) == np.str else x
        dataframe.rate = dataframe.rate.apply(remove_slash).str.strip().astype('float')
        return dataframe

    def table(self, dataframe):
        dataframe.online_order.replace(('Yes', 'No'), (True, False), inplace=True)
        dataframe.book_table.replace(('Yes', 'No'), (True, False), inplace=True)
        return dataframe

    def encoder(self, dataframe):
        encoder = LabelEncoder()
        dataframe = dataframe.apply(encoder.fit_transform)
        return dataframe

    def split(self, dataframe):
        X = dataframe.drop(['cost'], axis=1)
        y = dataframe['cost'].values
        return X, y
