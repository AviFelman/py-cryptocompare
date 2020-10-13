"""
This API wrapper is useful for gathering basic pricing information from the CryptoCompare API
There are a variety of transforms that are likely helpful for traders. 

Created by: @avifelman
Last Updated: Oct 8th, 2020 
"""

import requests
from datetime import date, datetime
import calendar
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from scipy import stats
import numpy as np

# For style purposes
plt.style.use('ggplot')


# Helper Function
def utc_to_datetime(df, formatting):
    if formatting == 'D':
        df_new = df.apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d'))
    if formatting == 'h':
        df_new = df.apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%S'))
    return df_new


# Create the class with all the pertinent parameters
class CryptoCompareAPI:
    def __init__(self, key):
        self.url = 'https://min-api.cryptocompare.com/data/'
        self.key = key

    # Create a helper function for performing requests.
    # There are few situations in which you will you need to call the API and access the non-json data
    def request(self, action, params):
        headers = {'authorization': 'Apikey %s' % self.key}
        response = requests.get(self.url + action, headers=headers, params=params)

        # Error handling
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
            if period == 'D':
                action = 'v2/histoday'
            elif period == 'h':
                action = 'v2/histohour'
            elif period == 'm':
                action = 'v2/histominute'
            else:
                print("Invalid interval")
            params = {'fsym': base, 'tsym': quote, 'limit': 20, 'toTs': date_to, 'e': exchange}
            data = self.request(action, params)
            return data['Data']

        if data_pull == 'e':
            if period == '1d':
                action = 'exchange/histoday'
            elif period == '1h':
                action = 'exchange/histohour'
            else:
                print("Invalid interval")
            params = {'tsym': quote, 'limit': 2000, 'toTs': date_to, 'e': exchange}
            data = self.request(action, params)
            return data

    # This function returns the current price of a single currency. Useful for ticker applications.
    def get_single_currency_price(self, base, quote_list):
        action = 'price'
        params = {'fsym': base, 'tsyms': quote_list}
        data = self.request(action, params)
        return data['Data']

    # This function returns either a DataFrame via the helper function.
    # Note that the date provided should always be in the form of a datetime object.
    # The API itself processes dates in UNIX format, so this function changes inputs to be UNIX based.
    def get_currency_history(self, base, period, date_from, date_to, quote='USD', exchange='CCCAGG'):
        # Create an empty DataFrame as a placeholder.
        df_main = pd.DataFrame()

        # Convert these dates into UNIX
        date_from = calendar.timegm(date_from.timetuple())
        date_to = calendar.timegm(date_to.timetuple())

        # Iterate through and populate the main DataFrame with data values.
        # This will request information in batches based on the earliest timestamp received.
        # The data pulls backwards (starting with the most recent, ending with the earliest)
        while date_from < date_to:
            data = self.get_data_helper(base, period, date_to, quote, exchange)
            df_main = df_main.append(data['Data'], ignore_index=True)
            date_to = data['TimeFrom']

        # The batches go back 2000 units every pull, so it can often overshoot the specified range.
        # This piece of logic fixes the overshoot
        df_main = df_main[df_main['time'] > date_from]

        # Transform DataFrame into a time-readable format
        df_main['time'] = utc_to_datetime(df_main['time'], period)

        # Create new headers and sort and set index. Also, dropping unnecessary columns.
        df_main.set_index('time', inplace=True)
        df_main.sort_index(ascending=True, inplace=True)
        df_main.drop(columns=['conversionType', 'conversionSymbol'], inplace=True)
        return df_main

    # This uses the opening price of currencies to create a DataFrame that contains the prices of many currencies
    # As opposed to more information about a specific currency like the get_currency_history function
    def get_multiple_currency_prices(self, currency_list, period, date_from, date_to, quote='USD', exchange='CCCAGG',
                                     returns=False, price='open'):
        
        if not isinstance(currency_list, list): 
            currency_list = [currency_list]
            
        df = pd.DataFrame()
        # Loop through all currencies in your currency list, to generate a DataFrame of multiple currencies.
        for currency in currency_list:
            df[currency] = self.get_currency_history(currency, period, date_from, date_to, quote, exchange)[price]
            # Set returns to True if you want a DataFrame of returns instead of prices
            if returns:
                df[currency] = df[currency].pct_change()[1:]
        return df

    # Returns the Alpha and the Beta of multiple assets against Bitcoin. 
    def get_alpha_beta(self, currency_list, period, date_from, date_to):
        data = self.get_multiple_currency_prices(currency_list, period, date_from, date_to, returns=True)[1:]
        btc_price = self.get_multiple_currency_prices(["BTC"], period, date_from, date_to, returns=True)[1:]
        df_real = pd.DataFrame()
        for column in data:
            (beta, alpha) = stats.linregress(btc_price.iloc[:, 0], data[column])[0:2]
            d = {"asset": [column], "alpha": [alpha], "beta": [beta]}
            df = pd.DataFrame(data=d)
            df_real = df_real.append(df)
        return df_real

    # This function generates a DataFrame with historical volatility of all coins, and plots accordingly. 
    # Can chart the volatiity of a variety of different assets
    def get_coin_volatility(self, currency_list, window, period, date_from, date_to, plot=True):
        data = self.get_multiple_currency_prices(currency_list, period, date_from, date_to, returns=True)
        i = 0
        for column in data:
            if is_numeric_dtype(data[column]):
                data[column] = data[column].rolling(window).std() * math.sqrt(365) * 100
                data.rename(columns={column: currency_list[i] + '_vol'}, inplace=True)
                i += 1
        if plot:        
            data.dropna().plot(linewidth=2.0)
            plt.title('{}{} Realized Volatility of Selected Coins'.format(window, period))
            plt.xticks(rotation=45)
            plt.ylabel('Volatility (%)')
            plt.show()
        return data

    # Returns the list of all CryptoCompare assets
    def get_toplist(self, limit):
        action = 'top/totalvolfull'
        params = {'limit': limit, 'tsym': 'USD'}
        data = self.request(action, params)['Data']
        toplist = []
        for item in data:
            toplist.append(item['CoinInfo']['Name'])
        return toplist

    # Hopefully self-explanatory
    def get_correlation_matrix(self, currency_list, period, date_from, date_to, plot=True):
        data = self.get_multiple_currency_prices(currency_list, period, date_from, date_to, returns=True)
        i = 0
        for column in data:
            if is_numeric_dtype(data[column]):
                data.rename(columns={column: currency_list[i]}, inplace=True)
                i += 1
        # Returns correlation matrix object
        correlation_object = data.corr()

        # Plotting the correlation object
        if plot:
            sns.heatmap(correlation_object, xticklabels=correlation_object.columns,
                        yticklabels=correlation_object.columns, annot=True)
            plt.title('{}D Correlations of Selected Coins'.format((date_to - date_from).days))
            plt.show()
        return correlation_object

    # TODO --> Add a function that takes in a specific asset, and then spits out it's top 5 most correlated assets,
    #  with a lag included Requires manual input because of enterprise access
    def get_top_correls(self, asset, period, date_from, date_to, lag=0):
        # need a manual list because of enterprise access
        
        if(lag > 0): 
            currency_list = ['BTC', 'ETH', 'LINK', 'XRP', 'YFI', 'LTC', 'TRX', 'ZEC', 'ATOM', 'COMP', 'ADA', 'BAND', 'BAL', 'REN', 'CELR', 'KAVA', 'SNX', 'RUNE', 'ANT']
            asset_px = self.get_multiple_currency_prices([asset], period, date_from, date_to, returns=True)[:-lag].reset_index().drop(columns='time')
            data = self.get_multiple_currency_prices(currency_list, period, date_from, date_to, returns=True)[lag:].reset_index().drop(columns='time')
            print(asset_px)
            print(data)
            df = asset_px.join(data, how='left')
            print(df)
            corrs = df.corr()
            return corrs.iloc[0].sort_values(ascending=False)[1:6]
        
        else: 
            currency_list = [asset, 'BTC', 'ETH', 'LINK', 'XRP', 'YFI', 'LTC', 'TRX', 'ZEC', 'ATOM', 'COMP', 'ADA', 'BAND', 'BAL','REN', 'CELR', 'KAVA', 'SNX', 'RUNE', 'ANT']
            data = self.get_multiple_currency_prices(currency_list, period, date_from, date_to, returns=True)
            data.dropna(inplace=True)
            corrs = data.corr()
            return corrs.iloc[0].sort_values(ascending=False)[1:6]

    # Compares the upside volatility of an asset to the downside volatility of an asset
    def directional_vol(self, asset, window, period, start_date, end_date):
        px_data = self.get_currency_history(asset, period, start_date, end_date)
        px_data['returns'] = px_data['close'].pct_change()
        px_data = px_data.assign(upside=np.where(px_data.returns > 0, px_data.returns, 0),
                                 downside=np.where(px_data.returns < 0, px_data.returns, 0))
        px_data['upside_vol'] = px_data.upside.rolling(window).std()
        px_data['downside_vol'] = px_data.downside.rolling(window).std(skipna=True)
        directional = px_data[['downside_vol', 'upside_vol']]
        fig, axs = plt.subplots(nrows=2, ncols=1)
        directional.plot(ax=axs[0], linewidth=2.0)
        px_data['close'].plot(ax=axs[1], linewidth=2.0)
        plt.show()
        return px_data

    # Delivers a historical volatility plot and the a graded percentile of volatility (from 0 to 100, every 5%)
    def vol_description(self, asset, window, period, date_from , date_to, graph): 
        vol = api.get_coin_volatility(asset, window, period, date_from, date_to, plot=graph)
        index = asset[0] + "_vol"
        curr_vol = vol[index].iloc[-1]
        vol_array = np.array([])

        for i in np.linspace(0,1,21): 
            vol_quantile = vol[index].quantile(i)
            vol_array = np.append(vol_array, vol_quantile)

            if vol_quantile > curr_vol > vol_array[len(vol_array)-2]: 
                print("{}%: {} vol <--- You are here, {} vol".format(round(i*100, 2), round(vol_quantile, 2), round(curr_vol, 2)))
            else: 
                print("{}%: {} vol".format(round(i*100, 2), round(vol_quantile, 2)))


if __name__ == '__main__':
    api_key = ''
    api = CryptoCompareAPI(api_key)
    date_from = datetime(2020, 10, 1)
    date_to = datetime.now()
    

    top_currencies = api.get_top_correls('LEND', period='D', date_from=date_from, date_to=date_to, lag=0)
    print(top_currencies)


    #api.get_coin_volatility(['BTC'], 30, 'D', date_from, date_to)
    #api.vol_description('SNX', 7, 'D', date_from, date_to, True)

    # Get Hourly Bitcoin Data going back as long as you want!
    #btc_data = api.get_currency_history('BTC', 'h', date_from, date_to)
    #btc_data.to_csv("hourly_btc_data.csv")


    # Example of Correlation Function
    #defi_list = ['MKR', 'ETH', 'ZRX', 'REP', 'KNC', 'COMP', 'YFI', 'SOL', 'OMG', 'BNT', 'LINK', 'BAND', 'BAL', 'REN', 'CELR', 'KAVA', 'LEND', 'SNX', 'RUNE', 'ANT']
    #api.get_correlation_matrix(curr_list, 'D', date_from, date_to)

  

    