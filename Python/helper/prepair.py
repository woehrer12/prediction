import csv
import json
import multiprocessing
import time
import datetime

import numpy as np
import pandas as pd
import talib.abstract as ta
from joblib import dump, load

import helper.binance
import helper.config


def compute_sma(df, window, colname):
    '''Computes Simple Moving Average column on a dataframe'''
    df[colname] = df['close'].rolling(window=window, center=False).mean()
    return(df)

def compute_rsi(df, window, colname):
    '''Computes RSI column for a dataframe. http://stackoverflow.com/a/32346692/3389859'''
    series = df['close']
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    # first value is sum of avg gains
    u[u.index[window - 1]] = np.mean(u[:window])
    u = u.drop(u.index[:(window - 1)])
    # first value is sum of avg losses
    d[d.index[window - 1]] = np.mean(d[:window])
    d = d.drop(d.index[:(window - 1)])
    rs = u.ewm(com=window - 1,ignore_na=False,
               min_periods=0,adjust=False).mean() / d.ewm(com=window - 1, ignore_na=False,
                                            min_periods=0,adjust=False).mean()
    df[colname] = 100 - 100 / (1 + rs)
    df[colname].fillna(df[colname].mean(), inplace=True)
    return(df)

def sort_all(CurrencyPairList):

    p = multiprocessing.Pool(12)

    for CurrencyPair in CurrencyPairList:

        p.apply_async(helper.prepair.sort_data, args=(CurrencyPair,))
    p.close()
    p.join()


def sort_data(CurrencyPair):
    df = pd.read_csv('./KI/Data/CurrencyPair_{}/Data.csv'.format(CurrencyPair))

    df = sorting(df)

    df.to_csv('./KI/Data/CurrencyPair_{}/Sorted.csv'.format(CurrencyPair), index=False)

def sorting(df, predict = True):
    # Convert the UNIX time into the date
    df["Date"] = pd.to_datetime(df["Open time"], unit='ms')

    df['hour'] = df['Date'].dt.hour
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['quarter'] = df['Date'].dt.quarter
    df['month'] = df['Date'].dt.month
#     df['year'] = df['Date'].dt.year
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['dayofmonth'] = df['Date'].dt.day

    # Sort data by date
    df = df.sort_values(by = 'Date')

    df.drop("Date", axis=1, inplace=True)

    # Use only the important values
    # df = df[['close', 'Date', 'high', 'low', 'open', 'volume']]

    #df["Date"] = pd.to_datetime(df["Date"])
    df['close'] = pd.to_numeric(df['close'])
    df['open'] = pd.to_numeric(df['open'])
    df['Bodysize'] = df['close'] - df['open']
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['Shadowsize'] = df['high'] - df['low']
    #TODO evtl Reihe erhöhen und auf auswirkungen schauen
    for window in [3, 8, 21, 55, 144, 377]: # several Fibonacci numbers
        # SMA
        df = compute_sma(df, window, colname = 'sma_{}'.format(window))
        # RSI
        df = compute_rsi(df, window, colname = 'rsi_{}'.format(window))
        # Values
        df["Min_{}".format(window)] = df["low"].rolling(window).min()
        df["Max_{}".format(window)] = df["high"].rolling(window).max()
        df["volume_{}".format(window)] = df["volume"].rolling(window).mean()
        df['PercentChange_{}'.format(window)] = df['close'].pct_change(periods = window)
        df['RelativeSize_sma_{}'.format(window)] = df['close'] / df['sma_{}'.format(window)]
        df['Diff_{}'.format(window)] = df['close'].diff(window)

    # Add modulo 10, 100, 1000, 500, 50
    df["Modulo_10"] = df["close"].copy() % 10
    df["Modulo_100"] = df["close"].copy() % 100
    df["Modulo_1000"] = df["close"].copy() % 1000
    df["Modulo_500"] = df["close"].copy() % 500
    df["Modulo_50"] = df["close"].copy() % 50

    # EMA - Exponential Moving Average
    df['ema3'] = ta.EMA(df, timeperiod=3)
    df['ema5'] = ta.EMA(df, timeperiod=5)
    df['ema10'] = ta.EMA(df, timeperiod=10)
    df['ema21'] = ta.EMA(df, timeperiod=21)
    df['ema50'] = ta.EMA(df, timeperiod=50)
    df['ema100'] = ta.EMA(df, timeperiod=100)

    # EMA Cross
    # 1 = BUY 2 = SELL 3 = HOLD
    df['ema3_21_cross'] = np.where((df['ema3'] > df['ema21']) & (df['ema3'].shift(1) <= df['ema21'].shift(1)), 1,
                            np.where((df['ema3'] < df['ema21']) & (df['ema3'].shift(1) >= df['ema21'].shift(1)), 2, 3))

    df['ema5_100_cross'] = np.where((df['ema5'] > df['ema100']) & (df['ema5'].shift(1) <= df['ema100'].shift(1)), 1,
                            np.where((df['ema5'] < df['ema100']) & (df['ema5'].shift(1) >= df['ema100'].shift(1)), 2, 3))
    

    # SMA - Simple Moving Average
    df['sma3'] = ta.SMA(df, timeperiod=3)
    df['sma5'] = ta.SMA(df, timeperiod=5)
    df['sma10'] = ta.SMA(df, timeperiod=10)
    df['sma21'] = ta.SMA(df, timeperiod=21)
    df['sma50'] = ta.SMA(df, timeperiod=50)
    df['sma100'] = ta.SMA(df, timeperiod=100)

    # SMA Cross
    # 1 = BUY 2 = SELL 3 = HOLD
    df['sma3_21_cross'] = np.where((df['sma3'] > df['sma21']) & (df['sma3'].shift(1) <= df['sma21'].shift(1)), 1,
                            np.where((df['sma3'] < df['sma21']) & (df['sma3'].shift(1) >= df['sma21'].shift(1)), 2, 3))

    df['sma5_100_cross'] = np.where((df['sma5'] > df['sma100']) & (df['sma5'].shift(1) <= df['sma100'].shift(1)), 1,
                            np.where((df['sma5'] < df['sma100']) & (df['sma5'].shift(1) >= df['sma100'].shift(1)), 2, 3))
    
    hilbert = ta.HT_SINE(df)
    df['htsine'] = hilbert['sine']
    df['htleadsine'] = hilbert['leadsine']

    # Parameter für Bollinger-Bänder
    N = 20  # Fenstergröße für den gleitenden Durchschnitt
    K = 2   # Faktor für die Standardabweichung

    # Berechnen Sie den gleitenden Durchschnitt (Mittlere Bollinger-Band)
    df['Middle_Band_BB'] = df['close'].rolling(window=N).mean()

    # Berechnen Sie die Standardabweichung
    df['Std_BB'] = df['close'].rolling(window=N).std()

    # Berechnen Sie die obere Bollinger-Bandlinie
    df['Upper_Band_BB'] = df['Middle_Band_BB'] + (K * df['Std_BB'])

    # Berechnen Sie die untere Bollinger-Bandlinie
    df['Lower_Band_BB'] = df['Middle_Band_BB'] - (K * df['Std_BB'])

    # TODO Add more indicators
    df['adx'] = ta.ADX(df)
    df['adxr'] = ta.ADXR(df)
    df['apo'] = ta.APO(df)
    aroon = ta.AROON(df)
    df['aroonup'] = aroon['aroonup']
    df['aroondown'] = aroon['aroondown']
    df['aroonosc'] = ta.AROONOSC(df)
    df['bop'] = ta.BOP(df)
    df['cci'] = ta.CCI(df)
    df['cmo'] = ta.CMO(df)
    df['dx'] = ta.DX(df)
    macd = ta.MACD(df)
    df['macd'] = macd['macd']
    df['macdsignal'] = macd['macdsignal']
    df['macdhist'] = macd['macdhist']
    df['mfi'] = ta.MFI(df)
    df['minus_di'] = ta.MINUS_DI(df)
    df['minus_dm'] = ta.MINUS_DM(df)
    df['mom'] = ta.MOM(df)
    df['plus_di'] = ta.PLUS_DI(df)
    df['plus_dm'] = ta.PLUS_DM(df)
    df['ppo'] = ta.PPO(df)
    df['roc'] = ta.ROC(df)
    df['rocp'] = ta.ROCP(df)
    df['rocr'] = ta.ROCR(df)
    df['rocr100'] = ta.ROCR100(df)
    df['rsi'] = ta.RSI(df)
    rsi = 0.1 * (df['rsi'] - 50)
    df['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)
    df['fisher_rsi_norma'] = 50 * (df['fisher_rsi'] + 1)
    stoch = ta.STOCH(df)
    df['slowd'] = stoch['slowd']
    df['slowk'] = stoch['slowk']
    stoch_fast = ta.STOCHF(df)
    df['fastd'] = stoch_fast['fastd']
    df['fastk'] = stoch_fast['fastk']
    stoch_rsi = ta.STOCHRSI(df)
    df['fastd_rsi'] = stoch_rsi['fastd']
    df['fastk_rsi'] = stoch_rsi['fastk']
    df['sar'] = ta.SAR(df)
    df['tema'] = ta.TEMA(df, timeperiod=9)
    df['trix'] = ta.TRIX(df)
    df['ultosc'] = ta.ULTOSC(df)
    df['willr'] = ta.WILLR(df)

    # Pattern Recognition - Bullish candlestick patterns
    # ------------------------------------
    # Hammer: values [0, 100]
    df['CDLHAMMER'] = ta.CDLHAMMER(df)
    # Inverted Hammer: values [0, 100]
    df['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(df)
    # Dragonfly Doji: values [0, 100]
    df['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(df)
    # Piercing Line: values [0, 100]
    df['CDLPIERCING'] = ta.CDLPIERCING(df) # values [0, 100]
    # Morningstar: values [0, 100]
    df['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(df) # values [0, 100]
    # Three White Soldiers: values [0, 100]
    df['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(df) # values [0, 100]

    # Pattern Recognition - Bearish candlestick patterns
    # ------------------------------------
    # Hanging Man: values [0, 100]
    df['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(df)
    # Shooting Star: values [0, 100]
    df['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(df)
    # Gravestone Doji: values [0, 100]
    df['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(df)
    # Dark Cloud Cover: values [0, 100]
    df['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(df)
    # Evening Doji Star: values [0, 100]
    df['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(df)
    # Evening Star: values [0, 100]
    df['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(df)

    # Pattern Recognition - Bullish/Bearish candlestick patterns
    # ------------------------------------
    # Three Line Strike: values [0, -100, 100]
    df['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(df)
    # Spinning Top: values [0, -100, 100]
    df['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(df) # values [0, -100, 100]
    # Engulfing: values [0, -100, 100]
    df['CDLENGULFING'] = ta.CDLENGULFING(df) # values [0, -100, 100]
    # Harami: values [0, -100, 100]
    df['CDLHARAMI'] = ta.CDLHARAMI(df) # values [0, -100, 100]
    # Three Outside Up/Down: values [0, -100, 100]
    df['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(df) # values [0, -100, 100]
    # Three Inside Up/Down: values [0, -100, 100]
    df['CDL3INSIDE'] = ta.CDL3INSIDE(df) # values [0, -100, 100]

    df["RSI"] = ta.RSI(df["close"], timeperiod = 14)
    df["ROC"] = ta.ROC(df["close"], timeperiod = 10)
    df["%R"] = ta.WILLR(df["high"], df["low"], df["close"], timeperiod = 14)
    df["OBV"] = ta.OBV(df["close"], df["volume"])
    df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = ta.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)


    # ATR - Average True Range
    atr_period = 14  # Anpassen Sie den Zeitraum nach Bedarf
    df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)


    # Definieren der Schwellenwerte für den Anstieg
    threshold_up = 1.05  # 5% Anstieg

    if predict == True:
        # Berechnen des Anstiegs basierend auf der "close"
        df["Prediction"] = df["close"].shift(-100)

    # Remove na values
    df.dropna(inplace=True)

    return df