import io
import configparser
import json
import pickle
from os import environ, path
from tempfile import TemporaryFile
import boto3

from pyspark import SparkConf
from pyspark.sql import SparkSession
from clustering_ex.spark_log4j import Log4j

s3 = boto3.client('s3')


def _load_settings(kwargs):
    settings = {}
    with open('./settings.json') as f:
        settings = json.load(f)

    for key in kwargs:
        if kwargs.get(key):
            settings[key] = kwargs[key]

    if not settings.get('base_bucket'):
        raise Exception("no target s3 path")
    return settings


def _s3_credential(session: SparkSession):
    config = configparser.ConfigParser()
    config.read(path.expanduser("~/.aws/credentials"))
    access_id = config.get('default', "aws_access_key_id")
    access_key = config.get('default', "aws_secret_access_key")
    hadoop_conf = session._jsc.hadoopConfiguration()
    hadoop_conf.set("fs.s3a.access.key", access_id)
    hadoop_conf.set("fs.s3a.secret.key", access_key)


def get_spark_app_config(configs: dict):
    spark_conf = SparkConf()

    for key, val in configs.items():
        spark_conf.set(key, val)
    return spark_conf


def start_spark(**kwargs):
    """[summary]
    jar_packages=[], files=[],

    Args:
        jar_packages (list, optional): [description]. Defaults to [].
        files (list, optional): [description]. Defaults to [].

    Returns:
        [type]: [description]
    """
    # detect execution environment
    flag_debug = 'DEBUG' in environ.keys()

    settings = _load_settings(kwargs)

    spark_builder = SparkSession.builder

    print('getting spark session')

    spark_conf = get_spark_app_config(settings['spark_app_configs'])
    spark_builder.config(conf=spark_conf)

    # create session and retrieve Spark logger object
    spark_session = spark_builder.getOrCreate()
    spark_logger = Log4j(spark_session)

    if flag_debug:
        _s3_credential(spark_session)

    print('spark session created')

    return spark_session, spark_logger, settings


def save_model_s3(model, target_bucket, prefix: str = 'lda_model'):
    # make it easier to load for local runs
    if model.isDistributed():
        model = model.toLocal()
    model.save(f'{target_bucket}/{prefix}/')


def upload_lda_vis_data_s3(lda_vis_data, target_bucket: str, prefix: str = 'lda_vis'):
    bucket_name = target_bucket.split('/')[2]
    with TemporaryFile() as tmp_file:
        pickle.dump(lda_vis_data, tmp_file)
        tmp_file.seek(0)
        s3.upload_fileobj(tmp_file, bucket_name, f'{prefix}/stats_dict.pickle')


def read_lda_vis_data_s3(target_bucket: str, prefix: str = 'lda_vis') -> dict:
    bucket_name = target_bucket.split('/')[2]
    response = s3.get_object(
        Bucket=bucket_name, Key=f'{prefix}/stats_dict.pickle')
    return pickle.loads(response['Body'].read())
