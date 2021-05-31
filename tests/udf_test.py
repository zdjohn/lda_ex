import pytest
import pyspark.sql.functions as F
from clustering_ex import udf
from pyspark.ml.linalg import Vectors


def test_udf_merge_lists(spark_resource):
    data = [([['a', 'b'], ['b', 'c']],), ([['a', 'b']],)]
    df = spark_resource.createDataFrame(data).toDF('list_set')
    result = df.select(udf.merge_lists('list_set').alias('merged_set'))
    assert len(result.first()['merged_set']) == 3
    assert result.count() == 2


def test_term_to_word(spark_resource):
    data = [([2, 1, 0],)]
    df = spark_resource.createDataFrame(data).toDF('word_index')
    result = df.select(udf.term_to_word(
        'word_index', F.lit('a,b,c,d')).alias('words'))
    assert result.first()['words'] == ['c', 'b', 'a']
    assert result.count() == 1


def test_get_topic(spark_resource):
    data = [(Vectors.dense([0.1, 0.2, 0.3, 0.4]),)]
    df = spark_resource.createDataFrame(data).toDF('score_list')
    result = df.select(udf.get_topic('score_list').alias('score_topic_index'))
    assert result.first()['score_topic_index']['topic'] == 3
