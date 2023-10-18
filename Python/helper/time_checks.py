import helper.binance
import helper.prepair
import helper.request
import helper.train
import helper.telegramsend

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
                    #'PEPEUSDT',
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

    try:
        columns = ['Open time', 'open', 'high', 'low', 'close', 'volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']

        df = pd.DataFrame(data, columns=columns)

        df = helper.prepair.sorting(df, False)

        text = str(CurrencyPair) + " " + str(timeframe) + " " + str(df['close'].iloc[-1]) + " \n" 


        BUY = 0
        SELL = 0

        ## RSI

        if df['rsi'].iloc[-1] < 20:        
            text = text + "RSI: " + str(df['rsi'].iloc[-1]) + " 🟢 \n"
            BUY += 1

        if df['rsi'].iloc[-1] > 80:
            text = text + "RSI: " + str(df['rsi'].iloc[-1]) + " 🔴 \n"
            SELL += 1

        ## SMA

        # 1 = BUY 2 = SELL 3 = HOLD

        SMA_BUY = 0
        SMA_SELL = 0

        if df['sma3_21_cross'].iloc[-1] == 1:
            text = text + "SMA 3_21 cross  🟢 \n"
            SMA_BUY += 1

        if df['sma3_21_cross'].iloc[-1] == 2:
            text = text + "SMA 3_21 cross  🔴 \n"
            SMA_SELL += 1

        if df['sma5_100_cross'].iloc[-1] == 1:
            text = text + "SMA 5_100 cross  🟢 \n"
            SMA_BUY += 1

        if df['sma5_100_cross'].iloc[-1] == 2:
            text = text + "SMA 3_21 cross  🔴 \n"
            SMA_SELL += 1


        if SMA_BUY > SMA_SELL:
            if SMA_BUY >= 2:
                BUY += 1
        else:
            if SMA_SELL >= 2:
                SELL += 1


        ## EMA

        # 1 = BUY 2 = SELL 3 = HOLD

        EMA_BUY = 0
        EMA_SELL = 0

        if df['ema3_21_cross'].iloc[-1] == 1:
            text = text + "EMA 3_21 cross  🟢 \n"
            EMA_BUY += 1

        if df['ema3_21_cross'].iloc[-1] == 2:
            text = text + "EMA 3_21 cross  🔴 \n"
            EMA_SELL += 1

        if df['ema5_100_cross'].iloc[-1] == 1:
            text = text + "EMA 5_100 cross  🟢 \n"
            EMA_BUY += 1

        if df['ema5_100_cross'].iloc[-1] == 2:
            text = text + "EMA 3_21 cross  🔴 \n"
            EMA_SELL += 1


        if EMA_BUY > EMA_SELL:
            if EMA_BUY >= 2:
                BUY += 1
        else:
            if EMA_SELL >= 2:
                SELL += 1


        ## MACD

        if df['macd'].iloc[-1] > df['macdsignal'].iloc[-1]:
            text = text + "MACD  🟢 \n"
            BUY += 1

        if df['macd'].iloc[-1] < df['macdsignal'].iloc[-1]:
            text = text + "MACD  🔴 \n"
            SELL += 1


        ## Stochastic

        STOCH_BUY = 0
        STOCH_SELL = 0

        # Signalüberprüfung für den langsam (slow) Stochastic Oscillator
        if df['slowk'].iloc[-1] > df['slowd'].iloc[-1]:
            text = text + "Slow K bigger Slow D  🟢 \n"
            STOCH_BUY += 1

        if df['slowk'].iloc[-1] < df['slowd'].iloc[-1]:
            text = text + "Slow K smaller Slow D  🔴 \n"
            STOCH_SELL += 1

        # Signalüberprüfung für den schnellen (fast) Stochastic Oscillator
        if df['fastk'].iloc[-1] > df['fastd'].iloc[-1]:
            text = text + "Fast K bigger Fast D  🟢 \n"
            STOCH_BUY += 1

        if df['fastk'].iloc[-1] < df['fastd'].iloc[-1]:
            text = text + "Fast K smaller Fast D  🔴 \n"
            STOCH_SELL += 1

        # Signalüberprüfung für den Stochastic RSI
        if df['fastk_rsi'].iloc[-1] > df['fastd_rsi'].iloc[-1]:
            text = text + "Fast K RSI bigger Fast D RSI  🟢 \n"
            STOCH_BUY += 1

        if df['fastk_rsi'].iloc[-1] < df['fastd_rsi'].iloc[-1]:
            text = text + "Fast K RSI smaller Fast D RSI  🔴 \n"
            STOCH_SELL += 1

        if STOCH_BUY > STOCH_SELL:
            if STOCH_BUY >= 2:
                BUY += 1
        else:
            if STOCH_SELL >= 2:
                SELL += 1

        ## Boolinger Bands
        # Überprüfen Sie, ob der Schlusskurs über dem unteren Bollinger-Band liegt.
        if df['close'].iloc[-1] < df['Lower_Band_BB'].iloc[-1]:
            text = text + "Close smaller Lower BB Band  🟢 \n"
            BUY += 1

        # Überprüfen Sie, ob der Schlusskurs unter dem oberen Bollinger-Band liegt.
        if df['close'].iloc[-1] > df['Upper_Band_BB'].iloc[-1]:
            text = text + "Close bigger Upper BB Band  🔴 \n"
            SELL += 1

        ## Check big change

        if df['close'].iloc[-1] < df['close'].iloc[-5]*0.95:
            text = text + "Big change  ⤵️🟢 \n"
            BUY += 1

        if df['close'].iloc[-1] > df['close'].iloc[-5]*1.05:
            text = text + "Big change  ⤴️🔴 \n"
            SELL += 1


        ## Prediction TODO

        # helper.train.predict(CurrencyPair, df)

        if BUY > SELL:
            if BUY >= 3:
                time_now = ((int(time.time())*1000))

                # Punkt 2: Volatilität basiert - ATR Multiplikator
                atr_multiplier = 2  # Anpassen Sie den Multiplikator nach Bedarf

                # Berechnung des Take-Profits basierend auf der Volatilität (ATR-Multiplikator)
                take_profit_volatility = df['close'].iloc[-1] + (atr_multiplier * df['atr'].iloc[-1])

                take_profit_percentage = ((take_profit_volatility - df['close'].iloc[-1]) / df['close'].iloc[-1]) * 100

                text = text + "Take profit volatility: " + str(take_profit_volatility) + " " + str(take_profit_percentage) + "%\n"

                text = text + "BUY !!!"

                for i in range(BUY):
                    text = text + "🚀"

                print("Aktuelle Zeit: " + str(datetime.datetime.fromtimestamp(time_now/1000)))
                print(text)
                helper.telegramsend.send(text)
        else:
            if SELL >= 3:
                time_now = ((int(time.time())*1000))

                # Punkt 2: Volatilität basiert - ATR Multiplikator
                atr_multiplier = 2  # Anpassen Sie den Multiplikator nach Bedarf

                # Berechnung des Take-Profits basierend auf der Volatilität (ATR-Multiplikator)
                take_profit_volatility = df['close'].iloc[-1] - (atr_multiplier * df['atr'].iloc[-1])

                take_profit_percentage = ((take_profit_volatility - df['close'].iloc[-1]) / df['close'].iloc[-1]) * 100

                text = text + "Take profit volatility: " + str(take_profit_volatility) + " " + str(take_profit_percentage) + "%\n"

                text = text + "SELL !!!"
                print("Aktuelle Zeit: " + str(datetime.datetime.fromtimestamp(time_now/1000)))
                print(text)

                for i in range(SELL):
                    text = text + "🔻"

                helper.telegramsend.send(text)

    
    except Exception as e:
        logging.error("Error while checks " + CurrencyPair + ": " + str(e))
        print("Error while checks " + CurrencyPair + ": " + str(e))


def check_5m():
    timeframe = "5m"

    # CurrencyPairList = helper.request.pairs()

    for CurrencyPair in CurrencyPairList:

        data = helper.binance.get_klines_5m(CurrencyPair)
        
        checks(data, CurrencyPair, timeframe)


def check_15m():
    timeframe = "15m"

    # CurrencyPairList = helper.request.pairs()

    for CurrencyPair in CurrencyPairList:

        data = helper.binance.get_klines_15m(CurrencyPair)
        
        checks(data, CurrencyPair, timeframe)


def check_1h():
    timeframe = "1h"

    # CurrencyPairList = helper.request.pairs()

    for CurrencyPair in CurrencyPairList:

        data = helper.binance.get_klines_1h(CurrencyPair)
        
        checks(data, CurrencyPair, timeframe)


def check_6h():
    timeframe = "6h"

    # CurrencyPairList = helper.request.pairs()

    for CurrencyPair in CurrencyPairList:

        data = helper.binance.get_klines_6h(CurrencyPair)
        
        checks(data, CurrencyPair, timeframe)


def check_1d():
    timeframe = "1d"

    # CurrencyPairList = helper.request.pairs()

    for CurrencyPair in CurrencyPairList:

        data = helper.binance.get_klines_1d(CurrencyPair)
        
        checks(data, CurrencyPair, timeframe)