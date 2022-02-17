# Docker compose down
# ==========================
docker-compose down --volumes
sudo unmount /dev/sda1 /media/avc/WD
sudo rm -rf /media/avc/WD/data/
sudo mount /dev/sda1 /media/avc/WD
