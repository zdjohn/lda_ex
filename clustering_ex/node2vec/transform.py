from typing import Tuple, List
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import IDF, StopWordsRemover, CountVectorizer, RegexTokenizer, NGram
from pyspark.ml.clustering import LDA, LocalLDAModel, LDAModel
from clustering_ex import udf


def to_item_df(df: DataFrame) -> DataFrame:
    """[summary]

    Args:
        df (DataFrame): raw json data frame
        root
        |-- availability: string (nullable = true)
        |-- brand: string (nullable = true)
        |-- e_brand_formatted: string (nullable = true)
        |-- e_brand_formatted_slug: string (nullable = true)
        |-- e_color: string (nullable = true)
        |-- e_color_parent: string (nullable = true)
        |-- e_image_urls_square_jpg: array (nullable = true)
        |    |-- element: array (containsNull = true)
        |    |    |-- element: string (containsNull = true)
        |-- e_matched_tokens_categories_formatted: array (nullable = true)
        |    |-- element: string (containsNull = true)
        |-- e_material: string (nullable = true)
        |-- e_price: double (nullable = true)
        |-- e_product_name: string (nullable = true)
        |-- gender: string (nullable = true)
        |-- id: string (nullable = true)
        |-- item_code: string (nullable = true)
        |-- long_description: string (nullable = true)
        |-- product_name: string (nullable = true)
        |-- retailer_code: string (nullable = true)
        |-- retailer_price: double (nullable = true)
        |-- retailer_url: string (nullable = true)

    Returns:
        DataFrame: product based data frame
        root
        |-- product_name: string (nullable = true)
        |-- codes: array (nullable = true)
        |    |-- element: string (containsNull = false)
        |-- brands: array (nullable = true)
        |    |-- element: string (containsNull = false)
        |-- retailers: array (nullable = true)
        |    |-- element: string (containsNull = false)
        |-- categories: array (nullable = true)
        |    |-- element: string (containsNull = true)
        |-- description: string (nullable = true)
        |-- price: double (nullable = true)
        |-- purchase_count: long (nullable = false)
        |-- retrailer_count: integer (nullable = false)
        |-- category_count: integer (nullable = false)
        |-- brand_count: integer (nullable = false)
        |-- code_count: integer (nullable = false)
        """

    return df.groupby(['product_name']).agg(
        F.collect_set('item_code').alias('codes'),
        F.collect_set('e_brand_formatted_slug').alias('brands'),
        F.collect_set('retailer_code').alias('retailers'),
        udf.merge_lists(F.collect_set('e_matched_tokens_categories_formatted')).alias(
            'categories'),
        F.collect_set('long_description')[0].alias('description'),
        F.mean('e_price').alias('price'),
        F.count('item_code').alias('purchase_count'),
    ).withColumn('retrailer_count', F.expr('size(retailers)')
                 ).withColumn('category_count', F.expr('size(categories)')
                              ).withColumn('brand_count', F.expr('size(brands)')
                                           ).withColumn('code_count', F.expr('size(codes)'))
