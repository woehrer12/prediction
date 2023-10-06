import helper.binance
import helper.prepair
import helper.request

import time
import datetime
import logging

import pandas as pd

CurrencyPairList = ['BTCUSDT',
                    'ETHUSDT', 
                    'AAVEUSDT', 
                    'ADAUSDT', 
                    'APEUSDT',
                    'ARBUSDT',
                    'ATOMUSDT',
                    'AVAXUSDT',
                    'BNBUSDT',
                    'DOGEUSDT',
                    'DOTUSDT',
                    'EGLDUSDT',
                    'ICPUSDT',
                    'IMXUSDT',
                    'KEYUSDT',
                    'LINKUSDT',
                    'LTCUSDT',
                    'MATICUSDT',
                    'PEPEUSDT',
                    'RNDRUSDT',
                    'SNXUSDT',
                    'SOLUSDT',
                    'STORJUSDT',
                    'SUIUSDT',
                    'TRXUSDT',
                    'WOOUSDT',
                    'XLMUSDT',
                    'XMRUSDT',
                    'XRPUSDT',
                    ]


def checks(data, CurrencyPair,timeframe):

    # try:

        columns = ['Open time', 'open', 'high', 'low', 'close', 'volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']

        df = pd.DataFrame(data, columns=columns)

        df = helper.prepair.sorting(df)

        BUY = 0
        SELL = 0

        ## RSI

        if df['rsi'].iloc[-1] > 80:
            # print("BUY ", CurrencyPair, " ", timeframe , ": RSI = ", df['rsi'].iloc[-1])
            BUY += 1

        if df['rsi'].iloc[-1] < 20:
            # print("SELL ", CurrencyPair, " ", timeframe , ": RSI = ", df['rsi'].iloc[-1])
            SELL += 1

        ## SMA

        # 1 = BUY 2 = SELL 3 = HOLD

        SMA_BUY = 0
        SMA_SELL = 0

        if df['sma3_21_cross'].iloc[-1] == 1:
            # print("BUY ", CurrencyPair, " ", timeframe, ": SMA 3_5 ")
            SMA_BUY += 1

        if df['sma3_21_cross'].iloc[-1] == 2:
            # print("SELL ", CurrencyPair, " ", timeframe, ": SMA 3_5 ")
            SMA_SELL += 1

        if df['sma5_100_cross'].iloc[-1] == 1:
            # print("BUY ", CurrencyPair, " ", timeframe, ": SMA 5_10 ")
            SMA_BUY += 1

        if df['sma5_100_cross'].iloc[-1] == 2:
            # print("SELL ", CurrencyPair, " ", timeframe, ": SMA 5_10 ")
            SMA_SELL += 1


        if SMA_BUY > SMA_SELL:
            if SMA_BUY >= 2:
                BUY += 1
        else:
            if SMA_SELL >= 2:
                SELL += 1


        ## MACD

        if df['macd'].iloc[-1] > df['macdsignal'].iloc[-1]:
            # print("BUY ", CurrencyPair, " ", timeframe, ": MACD")
            BUY += 1

        if df['macd'].iloc[-1] < df['macdsignal'].iloc[-1]:
            # print("SELL ", CurrencyPair, " ", timeframe, ": MACD")
            SELL += 1


        ## Stochastic

        STOCH_BUY = 0
        STOCH_SELL = 0

        # Signalüberprüfung für den langsam (slow) Stochastic Oscillator
        if df['slowk'].iloc[-1] > df['slowd'].iloc[-1]:
            # print("BUY ", CurrencyPair, " ", timeframe, ": Slow Stochastic Oscillator")
            STOCH_BUY += 1

        if df['slowk'].iloc[-1] < df['slowd'].iloc[-1]:
            # print("SELL ", CurrencyPair, " ", timeframe, ": Slow Stochastic Oscillator")
            STOCH_SELL += 1

        # Signalüberprüfung für den schnellen (fast) Stochastic Oscillator
        if df['fastk'].iloc[-1] > df['fastd'].iloc[-1]:
            # print("BUY ", CurrencyPair, " ", timeframe, ": Fast Stochastic Oscillator")
            STOCH_BUY += 1

        if df['fastk'].iloc[-1] < df['fastd'].iloc[-1]:
            # print("SELL ", CurrencyPair, " ", timeframe, ": Fast Stochastic Oscillator")
            STOCH_SELL += 1

        # Signalüberprüfung für den Stochastic RSI
        if df['fastk_rsi'].iloc[-1] > df['fastd_rsi'].iloc[-1]:
            # print("BUY ", CurrencyPair, " ", timeframe, ": Stochastic Oscillator RSI")
            STOCH_BUY += 1

        if df['fastk_rsi'].iloc[-1] < df['fastd_rsi'].iloc[-1]:
            # print("SELL ", CurrencyPair, " ", timeframe, ": Stochastic Oscillator RSI")
            STOCH_SELL += 1

        if STOCH_BUY > STOCH_SELL:
            if STOCH_BUY >= 2:
                BUY += 1
        else:
            if STOCH_SELL >= 2:
                SELL += 1

        ## Boolinger Bands
        # TODO

        if BUY > SELL:
            if BUY >= 3:
                print("BUY !!!", CurrencyPair, timeframe)
        else:
            if SELL >= 3:
                print("SELL !!!", CurrencyPair, timeframe)

    
    # except Exception as e:
    #     logging.error("Error while checks " + CurrencyPair + ": " + str(e))
    #     print("Error while checks " + CurrencyPair + ": " + str(e))


def check_5m():
    timeframe = "5m"

    time_now = ((int(time.time())*1000))
    print("Aktuelle Zeit: " + str(datetime.datetime.fromtimestamp(time_now/1000)))

    print("Time Check ", timeframe)

    CurrencyPairList = helper.request.pairs()

    for CurrencyPair in CurrencyPairList:

        data = helper.binance.get_klines_5m(CurrencyPair)
        
        checks(data, CurrencyPair, timeframe)


def check_15m():
    timeframe = "15m"

    time_now = ((int(time.time())*1000))
    print("Aktuelle Zeit: " + str(datetime.datetime.fromtimestamp(time_now/1000)))

    print("Time Check ", timeframe)

    # CurrencyPairList = helper.request.pairs()

    for CurrencyPair in CurrencyPairList:

        data = helper.binance.get_klines_15m(CurrencyPair)
        
        checks(data, CurrencyPair, timeframe)


def check_1h():
    timeframe = "1h"

    time_now = ((int(time.time())*1000))
    print("Aktuelle Zeit: " + str(datetime.datetime.fromtimestamp(time_now/1000)))

    print("Time Check ", timeframe)

    # CurrencyPairList = helper.request.pairs()

    for CurrencyPair in CurrencyPairList:

        data = helper.binance.get_klines_1h(CurrencyPair)
        
        checks(data, CurrencyPair, timeframe)