docker stop LidarAPI AIcam
sudo chmod 777 /dev/serial/by-id/usb-Prolific_Technology_Inc._USB-Serial_Controller-if00-port0
fuser -k /dev/serial/by-id/usb-Prolific_Technology_Inc._USB-Serial_Controller-if00-port0
docker-compose up --detach --remove-orphans