#!/bin/bash
if [ -d "/data/db" ]; then
    echo "Directory /data/db already exists."
    exit 0
fi
mkdir -p /data/db

# Start MongoDB server without sysctl
mongod --dbpath /data/db --fork --logpath /var/log/mongodb.log

# Run YCSB load and run mongodb commands
./ycsb-0.17.0/bin/ycsb load mongodb -s -P ycsb-0.17.0/workloads/workloada \
    -p recordcount=1000000 \
    -p mongodb.url=mongodb://localhost:27017/ycsb \
    -p mongodb.writeConcern=acknowledged
