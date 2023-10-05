## Steps to bring it running

# 1
First copy the ./config/config.ini.template to ./config/config.ini and insert your Binance API Key

# 2
Run ```./docker/prediction/compile.sh```

# 3
Run ```docker run -v ./:/prediction/ prediction```