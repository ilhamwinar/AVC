**GitHub Stat(s):**  
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/bhaktiyudha/AVC?logo=github) ![GitHub all releases](https://img.shields.io/github/downloads/ilhamwinar/AVC/total?logo=github)

**DockerHub Stat(s):**  
![Docker Pulls](https://img.shields.io/docker/pulls/yudhabhakti/avc-ai?logo=docker) ![Docker Image Size (tag)](https://img.shields.io/docker/image-size/yudhabhakti/avc-ai/latest?logo=docker) ![Docker Image Version (latest by date)](https://img.shields.io/docker/v/yudhabhakti/avc-ai?logo=docker&sort=date)

# Persiapan Control Unit
- ## Konfigurasi dan Instalasi Awal
  - Mendownload Repository
    ```bash
    git clone https://github.com/ilhamwinar/AVC
    ```
  - Melakukan Konfigurasi Awal
    ```bash
    sudo chmod +x avc_prepare.sh
    sudo ./avc_prepare.sh
    ```
  - Bila terdapat error /var/run/docker.sock
    ```bash
    sudo chmod 777 /var/run/docker.sock
    ```
    
  - Melakukan Download dan Update Aplikasi
    ```bash
    sudo ./update.sh
    ```
    
  - Mematikan dan menyalakan ulang Control Unit
    ```bash
    sudo reboot
    ```
- ## Parameter
  - Menulis parameter di file **.env**
    ```bash
    nano .env
    ``` 
    
   - buka environment.ini lalu copy ke **.env**, caranya bisa menggunakan sudo gedit parameter di file **.env** lalu copy kedalam file tersebut
    ```bash
    sudo gedit .env
    ```
  
    ```yaml
    #Control Unit Environment
    #================================================================
    #HOSTNAME --> Nama Gerbang dan Gardu (Penulisan "NamaGerbang"_"NomorGardu")
    #IP --> Alamat IP CU yang terhubung dengan akes Internet
    #================================================================
    HOSTNAME = CITEREUP2_1
    IP = 172.20.5.131

    #HDD Directory
    #================================================================
    #HDD --> Alamat direktori hardisk yang terbaca oleh CU
    #================================================================
    HDD = "/home/avc/data/"

    #GTO STATUS
    #================================================================
    #GTO --> kondisi "true" ketika terhubung dengan GTO sedangkan "false" tidak terhubung dengan GTO
    #================================================================
    GTO = "true"

    #Telegram
    #================================================================
    #TELEGRAM_BOT_TOKEN --> Token untuk BOT telegram
    #CHAT_ID --> Channel atau Grup ID Telegram Chat
    #GERBANG --> Nama Gerbang
    #GARDU --> Nomor Gardu
    #================================================================
    TELEGRAM_BOT_TOKEN = 1770185119:AAFKfQHyVHYwhwJT66IeL9mRiBl4nvW0UxQ
    CHAT_ID = -1001323845503
    GERBANG = CITEREUP2
    GARDU = 1

    #Dashboard Environment
    #================================================================
    #REACT_APP_PROXY3008 --> End Point Untuk Backend Lidar
    #REACT_APP_PROXY3007 --> End Point Untuk Backend Kamera
    #REACT_APP_PROXY3006 --> End Point Untuk Backend Audit
    #REACT_APP_PROXY8080 --> End Point Untuk Backend Image Viewer
    #================================================================
    REACT_APP_PROXY3008 = //172.20.5.131:3008
    REACT_APP_PROXY3007 = //192.168.100.168:3007
    REACT_APP_PROXY3006 = //172.20.5.131:3006
    REACT_APP_PROXY8080=//172.20.5.131:8080

    #AI Environment
    #================================================================
    # MODEL_12CAM --> model yang digunakan untuk mengklasifikasikan golongan kendaraan berdasarkan kamera 1 dan 2
    # MODEL_3CAM --> model yang digunakan untuk mengkalsifikasikan golongan kendaraan 4 dan 5 berdasarkan kamera 3
    # RTSP_CAM1, RTSP_CAM2,dan RTSP_CAM3 --> alamat RTSP untuk mengakses kamera 1,2,dan 3
    #================================================================
    MODEL_OBJECT_DETECTION_CAM12 = avc_cam12.engine
    MODEL_OBJECT_DETECTION_CAM3 = avc_cam3.engine
    RTSP_CAM1 = rtsp://root:avc12345@172.20.5.141/live1s1.sdp
    RTSP_CAM2 = rtsp://admin:avc12345@172.20.5.161/cam/realmonitor?channel=1&subtype=0
    RTSP_CAM3 = rtsp://admin:avc12345@172.20.5.181/cam/realmonitor?channel=1&subtype=0
    ```
# Aplikasi Control Unit
- Memperbarui aplikasi
  ```bash
    ./update.sh
  ```
- Membuat folder hardisk aplikasi
  ```bash
    ./makedir.sh
  ```
- Menjalankan aplikasi
  ```bash
    ./start.sh
  ```
- Menghentikan aplikasi
  ```bash
    ./stop.sh
  ```
