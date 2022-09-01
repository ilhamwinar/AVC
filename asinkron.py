import os
import asyncio
import logging
from decouple import config
from influxdb import InfluxDBClient
import requests
import time
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)

#environment
IP = config("IP")
TELEGRAM_BOT_TOKEN = config("TELEGRAM_BOT_TOKEN")
CHAT_ID = config("CHAT_ID")
GERBANG = config("GERBANG")
GARDU = config("GARDU")
# print(type(IP))
# print("---------------")
# print(IP)
# print("---------------")
# print(TELEGRAM_BOT_TOKEN)

gol_avc=[]
gol_etol=[]
gol_kosong=[0,0,0]
serial_error=[9,9]

# =============================================================
# Telegram Bot
# =============================================================
def telegram_bot_sendtext(bot_message):
    message = "AI Error di "+GERBANG+" "+GARDU + " "
    bot_token = TELEGRAM_BOT_TOKEN
    bot_chatID = CHAT_ID
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + \
        bot_chatID + '&parse_mode=Markdown&text=' + message + bot_message
    response = requests.get(send_text)
    return response.json()


#async def get_database():
def get_database():
    logging.info("STARTING QUERY ASINKRON DATA")
    time.sleep(60)
    while True:  
        n=0
        client = InfluxDBClient(host=IP,
                        port=8086,
                        username='',
                        password='',
                        database='avc')
        try:
            result_database = client.query('SELECT * FROM "jurnal_data" where time > now() - 8h order by desc limit 3 ')
            for measurement in result_database.get_points(measurement='jurnal_data'):
                if measurement["gol_avc"]==6:
                    measurement["gol_avc"]=1
                if measurement["gol_avc"]>8:
                    measurement["gol_avc"]=9
                if measurement["gol_etoll"]>8:
                    measurement["gol_etoll"]=9
                gol_avc.append(measurement["gol_avc"])
                gol_etol.append(measurement["gol_etoll"])

            if len(gol_avc) !=0 :
                logging.info("detail urutan AVC: "+str(gol_avc))
            else:
                logging.error("DATA AVC KOSONG")
            
            
            if len(gol_etol) !=0:
                logging.info("detail urutan ETOLL: "+str(gol_etol))
            else:
                logging.error("DATA ETOL KOSONG")

            client.close()
        except NameError as error:
            logging.error("DISCONNECT JETSON AND DATABASES") 
            time.sleep(60)
        except requests.exceptions.ConnectionError as e:
            logging.error("DISCONNECT JETSON AND DATABASES")  
            time.sleep(60)

        try:
            if (gol_avc[0]!=gol_etol[0]):
                n=n+1
            if (gol_avc[1]!=gol_etol[1]): 
                n=n+1
            if (gol_avc[2]!=gol_etol[2]) :
                n=n+1
            # if (gol_avc[3]!=gol_etol[3]) :
            #     n=n+1
            # if (gol_avc[4]!=gol_etol[4]) :
            #     n=n+1
            if 0 in gol_etol:
                logging.info("ada notran")
                try:
                    telegram_bot_sendtext("NOTRAN")
                except:
                    pass

        except IndexError as error:
            logging.error("DATA KOSONG SERIAL DISCONNECT")
            try:
                telegram_bot_sendtext("Data Kosong Karena Serial Disconnect")
            except:
                pass
            time.sleep(500)

        logging.info('Logic:'+str(n))

        if gol_avc == serial_error:
            logging.error("Serial mengirim golongan random")
            try:
                telegram_bot_sendtext("Error Serial, cek Kabel Serial.")
            except:
                pass

        if gol_avc != gol_kosong:
            if (gol_avc != serial_error) and (gol_etol != serial_error) :
                if n ==3:
                    logging.info("Restart Karena Asinkron")
                    os.system("docker restart LidarAPI")
                    time.sleep(40)
                    os.system("docker restart LidarAPI")
                else:
                    logging.error("Error Kerusakan cam atau Serial tidak kirim golongan atau Lidar selalu on")
                    try:
                        telegram_bot_sendtext("Error Kerusakan cam atau Serial tidak kirim golongan atau Lidar selalu on, SEGERA START STOP PROGRAM.")
                    except:
                        pass
                    os.system("docker restart LidarAPI")
                    time.sleep(40)
                    os.system("docker restart LidarAPI")
                    time.sleep(900)
        

        time.sleep(10)
        gol_avc.clear()
        gol_etol.clear()


if __name__ == "__main__":
    get_database()
