"""
This API wrapper is meant to be used to access and use the CryptoCompare API in an easy to use manner.
"""

import requests
from datetime import date, datetime
import calendar
from pandas.io.json import json_normalize
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Create the class with all the pertinent parameters
class cryptocompareAPI(object):
    def __init__(self, key):
        self.url = 'https://min-api.cryptocompare.com/data/'
        self.key = key

    ## Create a helper function for performing requests. There are few situations in which you will you need to call the API and access the non-json data
    def request(self, action, params):
        headers = {'authorization': 'Apikey %s' % (self.key)}
        response = requests.get(self.url + action, headers=headers, params=params)
        # Raise exceptions if something goes wrong with the API or a specific call.
        if self.key is None:
            raise Exception('API key is empty')
        if response.status_code != 200:
            raise Exception("Error: " + str(response.status_code) + "\n" + response.text + "\nRequest: " + response.url)
        json = response.json()
        print(response.url)
        return json

    # This function is a helper function that requests data and parses by period / type
    def get_data_helper(self, base, period, date_to, quote='USD', exchange='CCCAGG', data_pull='c'):
        action = ''
        if data_pull == 'c':
            if (period == '1d'):
                action = 'v2/histoday'
            elif (period == '1h'):
                action = 'v2/histohour'
            elif (period == '1m'):
                action = 'v2/histominute'
            else:
                print("Invalid interval")
            params = {'fsym': base, 'tsym': quote, 'limit': 2000, 'toTs': date_to, 'e': exchange}
            data = self.request(action, params)
            return data['Data']

        if data_pull == 'e':
            if (period == '1d'):
                action = 'exchange/histoday'
            elif (period == '1h'):
                action = 'exchange/histohour'
            else:
                print("Invalid interval")
            params = {'tsym': quote, 'limit': 2000, 'toTs': date_to, 'e': exchange}
            data = self.request(action, params)
            return data


    def utc_to_datetime(self, dataframe, formatting):
        if (formatting == '1d'):
            dataframe = dataframe.apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d'))
        if (formatting == '1h'):
            dataframe = dataframe.apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%S'))
        return dataframe

    ## Todo

    # This function returns the current price of a single currency. Useful for ticker applications.
    def get_single_currency_price(self, base, quote_list):
        action = 'price'
        params = {'fsym': base, 'tsyms': quote_list}
        data = self.request(action, params)
        return data['Data']

    # This function returns either a DataFrame via the helper function. Note that the date provided should always be in the form of a datetime object.
    # The API itself processes dates in UNIX format, so this function changes inputs to be UNIX based.
    def get_currency_history(self, base, period, date_from, date_to, quote='USD', exchange='CCCAGG'):

        # Create an empty DataFrame as a placeholder.
        df_main = pd.DataFrame()

        # Convert these dates into UNIX
        date_from = calendar.timegm(date_from.timetuple())
        date_to = calendar.timegm(date_to.timetuple())
        print(date_to)

        # Iterate through and populate the main DataFrame with data values. This will request information in batches based on the earliest timestamp recieved.
        while(date_from < date_to):
            data = self.get_data_helper(base, period, date_to, quote, exchange)
            df_main = df_main.append(data['Data'], ignore_index=True)
            date_to = data['TimeFrom']

        # The batches go back 2000 units every pull, so it can often overshoot the specified range. This fixes the overshoot
        df_main = df_main[df_main['time'] > date_from]

        # Transform DataFrame into a time-readable format.
        df_main['time'] = self.utc_to_datetime(df_main['time'], period)

        # Create new headers and sort and set index. Also, dropping unnecessary columns.
        df_main.set_index('time', inplace=True)
        df_main.sort_index(ascending=True, inplace=True)
        df_main.drop(columns=['conversionType', 'conversionSymbol'], inplace=True)
        return df_main

    # Note that this uses the opening price of currencies to create a dataframe that contains the prices of many currencies, as opposed to more information about a specific currency like the get_currency_history function.
    def get_multiple_currency_prices(self, currency_list, period, date_from, date_to, quote='USD', exchange='CCCAGG', include_returns=None, price='open'):
        df = pd.DataFrame()
        ## Loop through all currencies in your currency list, to generate a dataframe of multiple currencies.
        for currency in currency_list:
            new_header = currency + "_open"
            df[new_header] = self.get_currency_history(currency, period, date_from, date_to, quote, exchange)[price]
            ## Keep inside loop -- add column for generating a pct return next to each price movement. Helpful for calculating indicies. Use include_returns header to activate
            if include_returns is not None:
                df[currency + "_return"] = df[new_header].pct_change()
        return df

    # Returns a DataFrame with exchanges and their respective volumes
    ## TODO Fix this function, currently does not work
    def get_exchange_data(self, exchange_list, date_from, date_to, average=None, quote='USD'):
        df = pd.DataFrame()

        # The API only takes in UNIX dates, so we need to transform from the inputted datetime object to UNIX time
        date_from_unix = calendar.timegm(date_from.timetuple())
        date_to_unix = calendar.timegm(date_to.timetuple())

        print(date_from_unix)
        print(date_to_unix)

        for exchange in exchange_list:
            print(exchange)
            new_header = str(exchange)
            df_temp = pd.DataFrame()
            while(date_from_unix < date_to_unix):
                r = self.get_data_helper('base', '1h', date_to_unix, quote=quote, exchange=exchange, data_pull='e')
                df_temp = df_temp.append(r['Data'], ignore_index=True)
                date_to_unix = r['TimeFrom']

            df_temp.set_index('time', inplace=True)
            df_temp.sort_index(ascending=True, inplace=True)
            print(df_temp)
            df[new_header] = df_temp['volume']
        return df

    # This function returns index pricing based on two inputs 1) weights and 2) a DataFrame of prices.
    # Weights must be in the same order as columns in the DataFrame for this function to work.


    def get_beta(self, currency_list, period, date_from, date_to, window):
        data = self.get_multiple_currency_prices(currency_list, period, date_from, date_to)
        btc_prices = self.get_currency_history("BTC", '1d', date_from, date_to)['open']
        covariance = None
        std_dev = None
        return data



    # This function generates a DataFrame with historical volatility of all coins. If plot = True, it will generate a nice graph to go along.
    # TODO fix this function to be date malleable
    def get_coin_volatility(self, currency_list, window, period, periods_back, plot=False):
        data = self.get_currency_prices(currency_list, period, periods_back)
        i = 0
        for column in data:
            if (is_numeric_dtype(data[column])):
                data[column] = data[column].pct_change().rolling(window).std() * (365 ** 0.5)
                data.rename(columns={column: currency_list[i] + '_vol'}, inplace=True)
                i += 1
        if (plot):
            data.plot(x='time')
            plt.title('{}D Annualized Volatility of Selected Coins'.format(window))
            plt.show()
        return data

    ## TODO update correl function to produce top 5 correlations overtime from the top 30 assets on cryptocompare
    def get_correlation_matrix(self, currency_list, period, date_from, date_to, plot=False):
        data = self.get_multiple_currency_prices(currency_list, period, date_from, date_to)
        i = 0
        for column in data:
            if (is_numeric_dtype(data[column])):
                data.rename(columns={column: currency_list[i]}, inplace=True)
                i += 1
        # Returns correlation matrix object
        correlation_object = data.corr()

        # Plotting the correlation object
        if (plot):
            sns.heatmap(correlation_object, xticklabels=correlation_object.columns,
                        yticklabels=correlation_object.columns, annot=True)
            plt.title('{}D Correlations of Selected Coins'.format((date_to-date_from).days))
            plt.show()
        return correlation_object



if __name__ == '__main__':
    api_key = ''
    cryptocompare = cryptocompareAPI(api_key)
    date_from = datetime(2019, 1, 1)
    date_to = datetime.now()
    #data = cryptocompare.get_currency_history('BTC', '1h', date_from, date_to, quote='USD', exchange='CCCAGG')
    #data.to_csv("BTC_Hourly.csv")
    #df1 = cryptocompare.get_beta(['ETH', 'XRP', 'EOS'], '1d', date_from, date_to)
    correls = cryptocompare.get_correlation_matrix(['BTC', 'KNC', 'XTZ', 'LINK'], '1d', date_from, date_to, plot=True)
    #print(df1)
    # Want to analyze portfolio (based on amt of coins, the correlation of all coins in the portfolio, correlation of specific trades, etc
