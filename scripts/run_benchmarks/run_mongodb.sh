#!/bin/bash
./bin/ycsb run mongodb -s -P workloads/workloada \
    -p operationcount=1000000 \
    -p mongodb.url=mongodb://localhost:27017/ycsb \
    -p readproportion=0.25 \
    -p updateproportion=0.75 \
    -p mongodb.writeConcern=acknowledged
