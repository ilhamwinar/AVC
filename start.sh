# Docker compose up
# ==========================
docker-compose up --detach --remove-orphans

# Docker image prune
# ==================
docker image prune --all --force

sleep 5

# Docker restart LidarAPI
# ==================
docker restart LidarAPI
