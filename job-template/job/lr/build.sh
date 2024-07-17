#!/bin/bash

set -ex

docker build --network=host -t LinearRegression -f job/xgb/Dockerfile .
docker push LinearRegression