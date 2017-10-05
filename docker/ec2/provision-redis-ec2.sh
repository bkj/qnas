#!/bin/bash

# privision-redis-ec2.sh

sudo apt-get update
sudo add-apt-repository -y ppa:chris-lea/redis-server

sudo apt-get update
sudo apt-get install -y redis-server

redis-cli shutdown

sudo vi  /etc/redis/redis.conf
# 1 - Change port => port XXXX
# 2 - Add password => requirepass YYYYYYYYYYYYYYYYY
# 3 - Change bind to open up => comment out `# bind 127.0.0.1`

redis-cli start

# ... or something ... those `redis` commands might be incorrect
