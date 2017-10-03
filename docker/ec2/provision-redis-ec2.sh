#!/bin/bash

# privision-redis-ec2.sh

sudo apt-get update
sudo add-apt-repository -y ppa:chris-lea/redis-server
sudo apt-get update
sudo apt-get install -y redis-server

redis-cli shutdown

sudo vi  /etc/redis/redis.conf
# 1 - Change port => port 6879
# 2 - Add password => requirepass XXXXX
# 3 - Change bind => comment out `# bind 127.0.0.1`

