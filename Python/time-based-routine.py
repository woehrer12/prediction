import helper.functions
import helper.request
import helper.time_checks

import warnings
import schedule
import time

logger = helper.functions.initlogger("time-based-routine.log")

warnings.filterwarnings('ignore')

# Schedule alle 6 Stunden

schedule.every().hour.at(":00").do(helper.time_checks.check_5m)
schedule.every().hour.at(":05").do(helper.time_checks.check_5m)
schedule.every().hour.at(":10").do(helper.time_checks.check_5m)
schedule.every().hour.at(":15").do(helper.time_checks.check_5m)
schedule.every().hour.at(":20").do(helper.time_checks.check_5m)
schedule.every().hour.at(":25").do(helper.time_checks.check_5m)
schedule.every().hour.at(":30").do(helper.time_checks.check_5m)
schedule.every().hour.at(":35").do(helper.time_checks.check_5m)
schedule.every().hour.at(":40").do(helper.time_checks.check_5m)
schedule.every().hour.at(":45").do(helper.time_checks.check_5m)
schedule.every().hour.at(":50").do(helper.time_checks.check_5m)
schedule.every().hour.at(":55").do(helper.time_checks.check_5m)

schedule.every().hour.at(":00").do(helper.time_checks.check_15m)
schedule.every().hour.at(":15").do(helper.time_checks.check_15m)
schedule.every().hour.at(":30").do(helper.time_checks.check_15m)
schedule.every().hour.at(":45").do(helper.time_checks.check_15m)

schedule.every().hour.at(":00").do(helper.time_checks.check_1h)

schedule.every().day.at("00:00").do(helper.time_checks.check_6h)
schedule.every().day.at("06:00").do(helper.time_checks.check_6h)
schedule.every().day.at("12:00").do(helper.time_checks.check_6h)
schedule.every().day.at("18:00").do(helper.time_checks.check_6h)

schedule.every().day.at("00:00").do(helper.time_checks.check_1d)


print("START")

while True:
    schedule.run_pending()
    time.sleep(1)