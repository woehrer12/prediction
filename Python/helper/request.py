import csv
import logging
import os
import time
import multiprocessing


import helper.binance

SEQ_LEN = 100
DROPOUT = 0.2
WINDOW_SIZE = SEQ_LEN - 1

def pairs():
    CurrencyPairList = []

    pairs = helper.binance.get_exchange_info()

    for pair in pairs['symbols']:
        if 'USDT' in pair['symbol']:
            if 'SPOT' in pair['permissions']:
                if pair['status'] == 'TRADING':
                    if pair['symbol'].endswith("USDT"):
                        CurrencyPairList.append(pair['symbol'])
    return CurrencyPairList

def request_all(CurrencyPairList):

    p = multiprocessing.Pool(8)

    for CurrencyPair in CurrencyPairList:

        p.apply_async(helper.request.request, args=(CurrencyPair,))
        time.sleep(30)
    p.close()
    p.join()

def request(CurrencyPair):

    while True:
        try:
            logging.info("Request Data " + CurrencyPair)
            print("Request Data " + CurrencyPair)

            data = helper.binance.get_historical_klines_5m(CurrencyPair)

            # Start to Save Data
            # Open time / Open / High / Low / Close / Volume / Close time / Quote asset volume / Number of trades / Taker buy base asset volume / Taker buy quote asset volume / Ignore
            print("Save Data " + CurrencyPair)
            logging.info("Save Data " + CurrencyPair)
            if not os.path.exists('./KI/Data/CurrencyPair_{}'.format(CurrencyPair)):
                os.makedirs('./KI/Data/CurrencyPair_{}'.format(CurrencyPair))

            with open('./KI/Data/CurrencyPair_{}/Data.csv'.format(CurrencyPair), 'w', newline = '') as f:
                writer = csv.writer(f)
                writer.writerow(['Open time', 'open', 'high', 'low', 'close', 'volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
                for line in data:
                    writer.writerow(line)
            print("Finish Saved " + CurrencyPair)
            logging.info("Finish Saved " + CurrencyPair)
            break
        except Exception as e:
            logging.error("Error while request " + CurrencyPair + ": " + str(e))
            print("Error while request " + CurrencyPair + ": " + str(e))
            time.sleep(120)
