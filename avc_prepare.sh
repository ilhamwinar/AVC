#!/bin/bash
set -e

# Set up the timezone
# ===================
sudo timedatectl set-timezone Asia/Jakarta

# Set up the repository
# =====================
sudo apt-get update
sudo apt-get install curl -y

# Install Docker Engine
# =====================
SERVER_VERSION=$(docker version -f "{{.Server.Version}}")
SERVER_VERSION_MAJOR=$(echo "$SERVER_VERSION"| cut -d'.' -f 1)
SERVER_VERSION_MINOR=$(echo "$SERVER_VERSION"| cut -d'.' -f 2)
SERVER_VERSION_BUILD=$(echo "$SERVER_VERSION"| cut -d'.' -f 3)

if [ "${SERVER_VERSION_MAJOR}" -ge 20 ] && \
   [ "${SERVER_VERSION_MINOR}" -ge 10 ]  && \
   [ "${SERVER_VERSION_BUILD}" -ge 5 ]; then
    echo "Docker version >= 20.10.5 it's ok"
else
    echo "Docker version less than 20.10.5 can't continue"
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
fi

sudo chmod 777 /var/run/docker.sock
sudo usermod -aG docker $USER

# Install Docker Compose
# ======================
sudo curl -L "https://github.com/docker/compose/releases/download/v2.2.3/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Lazydocker
# ==================
sudo curl https://raw.githubusercontent.com/jesseduffield/lazydocker/master/scripts/install_update_linux.sh | bash

# Install sshpass
# ==================
sudo apt-get install sshpass 

# SwapMemory JetsonHacksNano
# ==================
sudo curl -o setSwapMemorySize.sh https://raw.githubusercontent.com/JetsonHacksNano/resizeSwapMemory/master/setSwapMemorySize.sh
sudo chmod +x setSwapMemorySize.sh
sudo ./setSwapMemorySize.sh -g 8

# Read HDD ExFat
# ==================
sudo add-apt-repository universe
sudo apt update 
sudo apt install exfat-fuse exfat-utils

# Setup PWM Fan
# ==================
sudo mv rc.local /etc/
sudo chmod u+x /etc/rc.local

# Modify serial port permission rules
# ===================================
echo -e "KERNEL==\"ttyS*\",MODE=\"0666\"\nKERNEL==\"ttyACM*\",MODE=\"0666\"\nKERNEL==\"ttyUSB*\",MODE=\"0666\"" | sudo tee /etc/udev/rules.d/99-serial.rules
sudo chmod 777 /dev/serial/by-id/usb-Prolific_Technology_Inc._USB-Serial_Controller-if00-port0

# Setup MTU for SSH by Linux
# ==================
sudo ifconfig eth0 mtu 1200

# Give Permission Start and Stop
# ==================
sudo chmod +x start.sh stop.sh update.sh makedir.sh mount.sh
