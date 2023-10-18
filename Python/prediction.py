import os
import warnings
import logging

import helper.functions
import numpy as np
import pandas as pd

import helper.binance
import helper.config
import helper.request
import helper.prepair
import helper.train

logger = helper.functions.initlogger("prediction.log")

warnings.filterwarnings('ignore')

conf = helper.config.initconfig()

if __name__ == "__main__":

    ## Create Data Folder
    if not os.path.exists('./KI'):
        os.makedirs('./KI')

    if not os.path.exists('./KI/Data'):
        os.makedirs('./KI/Data')

    if not os.path.exists('./KI/Predict'):
        os.makedirs('./KI/Predict')

    ## Search Pairs
    print("Search Pairs")

    # CurrencyPairList = helper.request.pairs()

    CurrencyPairList = ["BTCUSDT"] # TODO overwrite the List von Exchange



    ## Request Data
    print("Request Data")

    # helper.request.request_all(CurrencyPairList) # TODO


    ## Prepair Data

    # print("Prepair Data")

    # helper.prepair.sort_all(CurrencyPairList)

    for CurrencyPair in CurrencyPairList:
        print("Training")

        helper.train.train(CurrencyPair)