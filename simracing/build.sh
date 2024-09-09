#!/bin/bash

cd ..
pwd
sudo docker build -t sagol:sim -f simracing/Dockerfile .
