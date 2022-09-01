# Docker compose up
# ==========================
docker-compose up --detach --remove-orphans

# Docker image prune
# ==================
docker image prune --all --force

sleep 10
docker start LidarAPI
