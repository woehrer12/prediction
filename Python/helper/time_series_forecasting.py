#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import helper.binance

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def forecast(CurrencyPair):

    data = helper.binance.get_klines_5m(CurrencyPair)    

    columns = ['Open time', 'open', 'high', 'low', 'close', 'volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']

    df = pd.DataFrame(data, columns=columns)

    X = df.drop("close", axis=1)
    Y = df["close"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, shuffle= False)

    Y_train = pd.to_numeric(Y_train, errors='coerce')
    Y_test = pd.to_numeric(Y_test, errors='coerce')

    ARMAmodel = SARIMAX(Y_train, order = (1, 0, 1))
    ARMAmodel = ARMAmodel.fit()

    y_pred = ARMAmodel.get_forecast(len(Y_test.index))
    y_pred_df = y_pred.conf_int(alpha = 0.05) 
    y_pred_df["Predictions"] = ARMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
    y_pred_df.index = Y_test.index
    y_pred_out = y_pred_df["Predictions"] 
    plt.plot(y_pred_out, color='green', label = 'ARMA Predictions')
    plt.legend()

    arma_rmse = np.sqrt(mean_squared_error(Y_test.values, y_pred_df["Predictions"]))
    print("ARMA RMSE: ",arma_rmse)


    ARIMAmodel = ARIMA(Y_train, order = (5, 4, 2))
    ARIMAmodel = ARIMAmodel.fit()

    y_pred = ARIMAmodel.get_forecast(len(Y_test.index))
    y_pred_df = y_pred.conf_int(alpha = 0.05) 
    y_pred_df["Predictions"] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
    y_pred_df.index = Y_test.index
    y_pred_out = y_pred_df["Predictions"] 
    plt.plot(y_pred_out, color='Yellow', label = 'ARIMA Predictions')
    plt.legend()



    arma_rmse = np.sqrt(mean_squared_error(Y_test.values, y_pred_df["Predictions"]))
    print("ARIMA RMSE: ",arma_rmse)



    SARIMAXmodel = SARIMAX(Y_train, order = (5, 4, 2), seasonal_order=(2,2,2,12))
    SARIMAXmodel = SARIMAXmodel.fit()

    y_pred = SARIMAXmodel.get_forecast(len(Y_test.index))
    y_pred_df = y_pred.conf_int(alpha = 0.05) 
    y_pred_df["Predictions"] = SARIMAXmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
    y_pred_df.index = Y_test.index
    y_pred_out = y_pred_df["Predictions"] 
    plt.plot(y_pred_out, color='Blue', label = 'SARIMA Predictions')
    plt.legend()
    plt.savefig('./plot.png')


    arma_rmse = np.sqrt(mean_squared_error(Y_test.values, y_pred_df["Predictions"]))
    print("SARIMA RMSE: ",arma_rmse)