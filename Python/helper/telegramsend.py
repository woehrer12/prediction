import telegram
import telegram.constants
import helper.config
import logging
import asyncio

def send(text):
    asyncio.run(send2(text))

def send_image(image_path, caption=None):
    asyncio.run(send_image2(image_path, caption))

async def send2(text):
    conf = helper.config.initconfig()

    async with telegram.Bot(token=conf['telegramtoken']) as bot:
        try:
            await bot.send_message(conf['telegramid'], text=text, parse_mode=telegram.constants.ParseMode.HTML)
        except Exception as e:
            logging.error("Fehler beim senden des Telegram-Nachrichten: " + str(e) + "\n" + text)
            print("Fehler beim senden des Telegram-Nachrichten: " + str(e) + "\n" + text)


async def send_image2(image_path, caption=None):
    conf = helper.config.initconfig()

    async with telegram.Bot(token=conf['telegramtoken']) as bot:
        try:
            with open(image_path, 'rb') as image_file:
                await bot.send_photo(chat_id=conf['telegramid'], photo=image_file, caption=caption)
        except Exception as e:
            logging.error("Fehler beim Senden des Bildes über Telegram: " + str(e))
            print("Fehler beim Senden des Bildes über Telegram: " + str(e))
