import helper.functions
import helper.request
import helper.time_checks

import warnings
import schedule
import time

logger = helper.functions.initlogger("time-based-routine.log")

warnings.filterwarnings('ignore')

conf = helper.config.initconfig()

schedule.every().hour.at(":00").do(helper.time_checks.check_15m)
schedule.every().hour.at(":15").do(helper.time_checks.check_15m)
schedule.every().hour.at(":30").do(helper.time_checks.check_15m)
schedule.every().hour.at(":45").do(helper.time_checks.check_15m)
schedule.every().hour.at(":00").do(helper.time_checks.check_1h)


print("START")

while True:
    schedule.run_pending()
    time.sleep(1)