#!/bin/bash
mkdir -p /data/db

# Start MongoDB server without sysctl
mongod --dbpath /data/db --fork --logpath /var/log/mongodb.log

# Run YCSB load and run mongodb commands
./bin/ycsb load mongodb -s -P workloads/workloada \
    -p recordcount=1000000 \
    -p mongodb.url=mongodb://localhost:27017/ycsb \
    -p mongodb.writeConcern=acknowledged
