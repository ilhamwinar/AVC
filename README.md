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
    ```yaml
      #Control Unit Environment
      #================================================================
      #HOSTNAME --> Nama Gerbang dan Gardu (Penulisan "NamaGerbang"_"NomorGardu")
      #IP --> Alamat IP CU yang terhubung dengan akes Internet
      #================================================================
      HOSTNAME = CIKUNIR4_5
      IP = 10.0.3.2

      #HDD Directorys
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
      GERBANG = CIKUNIR4
      GARDU = 5

      #Dashboard Environment
      #================================================================
      #REACT_APP_PROXY3008 --> End Point Untuk Backend Lidar
      #REACT_APP_PROXY3007 --> End Point Untuk Backend Kamera
      #REACT_APP_PROXY3006 --> End Point Untuk Backend Audit
      #REACT_APP_PROXY8080 --> End Point Untuk Backend Image Viewer
      #================================================================
      REACT_APP_PROXY3008 = //10.0.3.2:3008
      REACT_APP_PROXY3007 = //192.168.100.168:3007
      REACT_APP_PROXY3006 = //10.0.3.2:3006
      REACT_APP_PROXY8080=//10.0.3.2:8080

      #AI Environment
      #================================================================
      # MASK_CAM1, MASK_CAM2,dan MASK_CAM3 -->  nama file untuk masking gambar camera 1,2,dan 3
      # MODEL_2CAM --> model yang digunakan untuk mengklasifikasikan golongan kendaraan berdasarkan kamera 1 dan 2
      # MODEL_3CAM --> model yang digunakan untuk mengkalsifikasikan golongan kendaraan 4 dan 5 berdasarkan kamera 3
      # RTSP_CAM1, RTSP_CAM2,dan RTSP_CAM3 --> alamat RTSP untuk mengakses kamera 1,2,dan 3
      #================================================================
      MASK_CAM1 = mask-6class.jpg
      MASK_CAM2 = mask-6class-cam2.jpg
      MASK_CAM3 = mask-6class-cam3.jpg
      MODEL_2CAM = 7class-v1.plan
      MODEL_3CAM = 6class-3cam-d1-bn-v2.plan
      MODEL_OBJECT_DETECTION_CAM12 = avc_cam12.engine
      MODEL_OBJECT_DETECTION_CAM3 = avc_cam3.engine
      RTSP_CAM1 = rtsp://root:avc12345@10.0.3.10/live1s1.sdp
      RTSP_CAM2 = rtsp://admin:avc12345@10.0.3.11/cam/realmonitor?channel=1&subtype=0
      RTSP_CAM3 = rtsp://admin:avc12345@10.0.3.12/cam/realmonitor?channel=1&subtype=0
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
