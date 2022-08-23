# Docker compose up
# ==========================
docker-compose up --detach --remove-orphans

# Docker image prune
# ==================
docker image prune --all --force

# Docker restart LidarAPI
# ==================
docker restart LidarAPI
