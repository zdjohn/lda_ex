# LDA clustering with mllib

## pre-requisite

- aws accounts
- credentials with emr/s3/airflow policy attached
- pyspark dev environment
- python,tox

## Setup dev environment

Create virtual dev environment via tox

RUN: `tox -e dev`

ACTIVATE: `:{your project root}$ source .tox/dev/bin/activate`

OPEN NOTEBOOK: `jupyter notebook`

Happy coding!

## Submit job to EMR

submit the job to emr is easy

RUN: `bash emr.sh`

Under the hood following things happened:

1. your python code including your settings will be packaged into `dist` folder as emr job artifacts
2. those artifacts then is uploaded to s3 you pointed to inside config
3. EMR cluster spinning, then run the etl job, and saves its output under `output` prefix

## what dose the job do

see the details in notebook:
