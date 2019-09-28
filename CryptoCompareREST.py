## Import all the of the necessary repositories for usage.
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from pandas.io.json import json_normalize
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns

## Create the class with all the pertinent paramaters
class cryptocompareAPI(object):
    def __init__(self, key):
        self.url = 'https://min-api.cryptocompare.com/data/'
        self.key = key

    ## Create a helper function for performing requests.
    ## In almost no situation will you need to call the API and access the non-json data
    def request(self, action, params):
        headers = {'authorization': 'Apikey %s' % (self.key)}
        response = requests.get(self.url + action, headers=headers, params=params)
        # Raise exceptions if something goes wrong with the API or a specific call.
        if self.key is None:
            raise Exception('API key is empty')
        if response.status_code != 200:
            raise Exception("Error: " + str(response.status_code) + "\n" + response.text + "\nRequest: " + response.url)
        json = response.json()
        return json

    def utc_to_datetime(self, dataframe, formatting):
        if(formatting == '1d'):
            dataframe = dataframe.apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d'))
        if(formatting == '1h'):
            dataframe = dataframe.apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%S'))
        return dataframe
    # Get a single currency -- currenct pricing
    def get_single_currency_price(self, base, quote_list):
        action = 'price'
        params =  {'fsym': base, 'tsyms': quote_list}
        data = self.request(action, params)
        return data['Data']

    def get_single_currency_history(self, base, period, periods_back, quote='USD'):
        action = ''
        if(period == '1d'):
            action = 'v2/histoday'
        elif(period == '1h'):
            action = 'v2/histohour'
        elif(period == '1m'):
            action = 'v2/histominute'
        else:
            print("Invalid interval")

        params = {'fsym': base, 'tsym': quote, 'limit': periods_back}
        data = self.request(action, params)
        return data['Data']['Data']

    ## This function creates a dataframe from the get_single_currency_price function that only contains a formatted date
    ## and corresponding price
    def helper_open_price_to_dataframe(self, base, period, periods_back,quote='USD'):
        new_header = base + "_open"
        data = json_normalize(self.get_single_currency_history(base, period, periods_back))

        if(period == '1d'):
            data['time'] = self.utc_to_datetime(data['time'], "1d")
        else:
            data['time'] = self.utc_to_datetime(data['time'], "1h")

        data[new_header] = data['open']
        return data[['time', new_header]]

    # Note that this uses the opening price of any day put into the function
    def get_currency_prices(self, currency_list, period, periods_back,quote='USD',include_returns=None):
        dataframe_final = pd.DataFrame()

        ## Loop for creating the DataFrame
        for currency in currency_list:
            new_header = currency + "_open"
            if(currency == currency_list[0]):
                dataframe_final = self.helper_open_price_to_dataframe(currency, period, periods_back)
            else:
                dataframe_final[new_header] = self.helper_open_price_to_dataframe(currency, period, periods_back)[new_header]
            ## Keep inside loop -- add column for generating a pct return next to each price movement. Helpful for calculating indicies. Use include_returns header to activate
            if(include_returns is not None):
                dataframe_final[currency + "_return"] = dataframe_final[new_header].pct_change()
        return dataframe_final

    # Returns a dataframe with exchanges and their respective volume graphs.
    # If you would like volumes to be averaged over a period of time to reduce noise, just change average to a int() of your choosing (7, 14, 21, etc)
    # If you would like to show a marketshare graph, change the parameter, percentages to True.
    def get_exchange_data(self, exchange_list, periods_back, quote='USD', average=None, percentages=False, plot=False):
        dataframe_final = pd.DataFrame()
        action = "exchange/histoday"

        for exchange in exchange_list:
            new_header = str(exchange)
            params = {'tsym': quote, 'limit': periods_back, 'e': exchange}

            if(exchange == exchange_list[0]):
                dataframe_final = json_normalize(self.request(action, params)["Data"])
                dataframe_final.rename(columns={'volume': new_header}, inplace=True)
            else:
                dataframe_final[new_header] = json_normalize(self.request(action, params)["Data"])['volume']

        dataframe_final['time'] = self.utc_to_datetime(dataframe_final['time'], '1d')
        dataframe_final.set_index('time', inplace=True)

        if average is not None:
            dataframe_final = dataframe_final.rolling(window=average).mean()

        if(plot):
            dataframe_final.plot()
            plt.title('{}D Moving Average of Exchange Volumes'.format(average))
            plt.show()

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

    # This function generates a dataframe with historical volatility of all coins. If plot = True, it will generate a nice graph to go along.
    def get_coin_volatility(self, currency_list, window, period, periods_back, plot=False):
        data = self.get_currency_prices(currency_list, period, periods_back)
        i = 0
        for column in data:
            if(is_numeric_dtype(data[column])):
                data[column] = data[column].pct_change().rolling(window).std()*(365**0.5)
                data.rename(columns={column: currency_list[i] + '_vol'}, inplace=True)
                i+=1
        if(plot):
            data.plot(x='time')
            plt.title('{}D Annualized Volatility of Selected Coins'.format(window))
            plt.show()
        return data

    # This function allows you to inset any list of currencies, and print a table of correlation data
    # As part of the function, Seaborn is included to create a nice looking graph. If you would like the graph, set plot = True.
    def get_correlation_matrix(self, currency_list, period, periods_back, plot=False):
        data = self.get_currency_prices(currency_list, period, periods_back)
        i = 0
        for column in data:
            if(is_numeric_dtype(data[column])):
                data.rename(columns={column: currency_list[i]}, inplace=True)
                i+=1
        # Returns correlation matrix object
        correlation_object = data.corr()

        # Plotting the correlation object
        if(plot):
            sns.heatmap(correlation_object, xticklabels=correlation_object.columns, yticklabels=correlation_object.columns, annot=True)
            plt.title('{}D Correlations of Selected Coins'.format(periods_back))
            plt.show()
        return correlation_object


if __name__ == '__main__':
    api_key = ''
    cryptocompare = cryptocompareAPI(api_key)

    ## Change list names to get different exchanges. Change 365 --> will change the amount of days back the data pulls.
    # changing average will change the moving average calculation (14 day moving average of volume to reduce spikes and spot a trend)
    cryptocompare.get_correlation_matrix(['BTC', 'ETH', 'XRP', 'ETC', 'TRX'], '1d', 365, plot=True)
