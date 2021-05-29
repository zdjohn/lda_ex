#!/bin/bash

aws s3 cp s3://zip-ex/artifact/requirements.txt .
sudo python3 -m pip install -r requirements.txt

export PYSPARK_PYTHON=python3:$PYSPARK_PYTHON
export PYSPARK_DRIVER_PYTHON=python3:$PYSPARK_DRIVER_PYTHON