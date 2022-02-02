# Update Repository
# ==========================
git pull

# Create Folder for Mount Volumes
# ==========================
sudo mkdir -p /media/avc/WD/data/{0,1,2,3,4,5,8,lidar}
sudo mkdir -p db 
sudo mkdir -p influxdb

# Docker compose pull
# ==========================
docker-compose pull