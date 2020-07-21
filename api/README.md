docker build -t willarth/attention-monitor-api .
docker push willarth/attention-monitor-api

docker run -d --name attention-monitor-api -p 80:80 -e AWS_ACCESS_KEY_ID=<key> -e AWS_SECRET_ACCESS_KEY=<key> willarth/attention-monitor-api

ssh -i "attention-monitor-key.pem" ubuntu@ec2-13-229-155-101.ap-southeast-1.compute.amazonaws.com