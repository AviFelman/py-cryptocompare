## Programming the CryptoCompare API Calls

# First -- I want to be able to construct the URL needed
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from pandas.io.json import json_normalize
import matplotlib as plt

class cryptocompareAPI(object):
    def __init__(self, key):
        self.url = 'https://min-api.cryptocompare.com/data/'
        self.key = key

    def request(self, action):
        headers = {'authorization': 'Apikey %s' % (api_key)}
        response = requests.get(self.url + action, headers=headers)
        # Catching errors
        if self.key is None:
            raise Exception('API key is empty')
        if response.status_code != 200:
            raise Exception("Error: " + str(response.status_code) + "\n" + response.text + "\nRequest: " + response.url)

        json = response.json()
        return json

    def get_raw_price_data(self, currency, days_back, additional_information=False):
        action = 'histoday?fsym={}&tsym=USD&limit={}'.format(currency, days_back)
        data = self.request(action)

        if(additional_information):
            return data
        return data['Data']

    def get_open_history(self, currency, days_back):
        new_header = currency + "_Open"
        data = json_normalize(self.get_raw_price_data(currency, days_back))
        data['time'] = data['time'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d'))
        data[new_header] = data['open']
        return data[['time', new_header]]

    def get_multiple_currencies(self, currency_list, days_back, data=None):
        dataframe_final = pd.DataFrame()
        for item in currency_list:
            new_header = item + "_Open"
            if(item == currency_list[0]):
                dataframe_final = self.get_open_history(item, days_back)
            else:
                dataframe_final[new_header] = self.get_open_history(item, days_back)[new_header]
            dataframe_final[item + "_return"] = dataframe_final[new_header].pct_change()

            if(data == 'returns'):
                dataframe_final.drop([new_header], axis=1, inplace=True)

        dataframe_final.set_index(['time'], inplace=True)
        return dataframe_final

    def get_exchange_data(self, exchange_list, days):
        dataframe_final = pd.DataFrame()
        for item in exchange_list:
            action = "exchange/histoday?tsym=USD&limit={}&e={}".format(days, item)
            new_header = str(item)
            if(item == exchange_list[0]):
                dataframe_final = json_normalize(self.request(action)["Data"])
                dataframe_final.rename(columns={'volume': new_header}, inplace=True)
            else:
                dataframe_final[new_header] = json_normalize(self.request(action)["Data"])['volume']
        dataframe_final['time'] = dataframe_final['time'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d'))
        return dataframe_final

    def index_data(self, weights, price_data):
        df = pd.DataFrame()
        i = 0
        for column in price_data:
            df[column+'_weighted']=price_data[column]*weights[i]
            i=i+1
        df['index_returns'] = df.sum(axis=1)
        df['index_values'] = (df['index_returns']+1).cumprod()
        return df





    # Each other endpoint will have URL specific endpoints.
    # First one should be just gathering



if __name__ == '__main__':
    api_key = '60a704b50ae660f09c6e053b1a5fa8027e95d145b3f21061e359d1cb185c5fe4'
    CryptoCompare = cryptocompareAPI(api_key)
    #june = ['BTC','ETH','XRP','BCH','EOS','LTC','BNB','BSV','XLM','ADA','TRX','MIOTA','ATOM','XTZ','ETC','NEO','XEM','MKR','BTG']
    #july = ['BTC','XRP','ETH','BCH','EOS','XLM','LTC','BSV','TRX','ADA','MIOTA','BNB','XEM','ETC','NEO','MKR','WAVES','XTZ','DOGE','BTG']
    #august = ['BTC','XRP','ETH','BCH','EOS','XLM','LTC','BSV','TRX','ADA','MIOTA','BNB','XEM','ETC','NEO','MKR','WAVES','XTZ','DOGE','BTG']

    #june_weights = [0.622,0.116,0.076,0.032,0.029,0.029,0.019,0.015,0.011,0.009,0.009,0.006,0.005,0.004,0.004,0.003,0.003,0.003,0.003,0.002]
    #july_weights = [0.684,0.11,0.06,0.027,0.025,0.019,0.016,0.008,0.008,0.007,0.006,0.004,0.004,0.004,0.004,0.003,0.003,0.003,0.002,0.002]
    #august_weights = [0.72,0.091,0.053,0.024,0.023,0.017,0.016,0.01,0.006,0.006,0.006,0.005,0.004,0.003,0.003,0.003,0.003,0.003,0.002,0.002]

    #exchange = CryptoCompare.get_exchange_data(['Binance', 'Coinbase', 'BitMEX', 'Bitfinex'], 100)


    #data = CryptoCompare.get_multiple_currencies(['BTC', 'ETH'], 1000)
    data = CryptoCompare.get_multiple_currencies(['BTC'], 1200)
    #return_data = CryptoCompare.index_data(august_weights, data)
    #index_data = return_data['time', 'index_values']
    #data.to_csv("index_data_january.csv")
    #print(data)
