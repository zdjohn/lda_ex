# package 
tox -e pack

# upload to s3 
aws s3 cp dist s3://zip-ex/artifact/ --recursive

#involke
aws emr create-cluster --name "lda-k5" \
    --release-label emr-6.1.0 \
    --applications Name=Spark \
    --log-uri s3://zip-ex/logs/ \
    --ec2-attributes KeyName=emr-sample-keypair \
    --instance-type c5.xlarge \
    --instance-count 3 \
    --configurations '[{"Classification":"spark-env","Configurations":[{"Classification":"export","Properties":{"PYSPARK_PYTHON":"/usr/bin/python3"}}]}]' \
    --bootstrap-action Path="s3://zip-ex/artifact/bootstrap.sh" \
    --steps Type=Spark,Name="lda-5",ActionOnFailure=TERMINATE_CLUSTER,Args=[--deploy-mode,cluster,--master,yarn,--py-files,s3://zip-ex/artifact/dist_files.zip,--files,s3://zip-ex/artifact/settings.json,s3://zip-ex/artifact/main.py,--k=7] \
    --use-default-roles \
    --auto-terminate