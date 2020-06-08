docker build -t attention-monitor-api .

docker run -d --name attention-monitor-api -p 80:80 attention-monitor-api
