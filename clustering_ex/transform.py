from typing import Tuple
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


def to_tf_idf_features_df(df: DataFrame) -> Tuple[list, DataFrame]:
    """
    clean up html markups as well as punctuation from description
    transform item description into tf-idf vectors

    Args:
        df (DataFrame): item based data frame

    Returns:
        DataFrame: feature engineered based on tf-idf vectorization
        root
        |-- product_name: string (nullable = true)
        |-- brands: array (nullable = true)
        |    |-- element: string (containsNull = false)
        |-- categories: array (nullable = true)
        |    |-- element: string (containsNull = true)
        |-- features: vector (nullable = true)
    """

    html_punctuation_re = r'<.*?>|[-\\/\\"_():;,.!?\\-\\*•\\|\\&\\$%]|[0-9]'

    cleaned = df.withColumn('description', F.regexp_replace(
        'description', html_punctuation_re, ''))

    tokenizer = RegexTokenizer(inputCol="description", outputCol="words")
    remover = StopWordsRemover(
        inputCol="words", outputCol="cleaned_words")
    # NOTE: optionally n_gram approach seems performs better when K value being larger
    # n_gram = NGram(n=2, inputCol="remove_stop_words", outputCol="cleaned_words")
    tf = CountVectorizer(inputCol="cleaned_words",
                         outputCol="raw", maxDF=0.3, minDF=0.01)
    idf = IDF(inputCol="raw", outputCol="features")

    # pipeline = Pipeline(stages=[tokenizer, remover, n_gram, tf, idf])
    pipeline = Pipeline(stages=[tokenizer, remover, tf, idf])

    features_pipeline = pipeline.fit(cleaned)
    feature_df = features_pipeline.transform(
        cleaned
    ).select('product_name', 'brands', 'categories', 'features', 'cleaned_words')

    vocab = features_pipeline.stages[2].vocabulary

    return vocab, feature_df


def fit_lda(df: DataFrame, k: int = 6, iter: int = 50, seed: int = 0, col: str = 'features') -> Tuple[LDAModel, DataFrame]:
    """
    fit dataframe with lda model

    Args:
        DataFrame ([type]): [description]
        df (DataFrame, k, optional): [description]. Defaults to 6, iter: int = 50, col: str = 'features')->(LocalLDAModel.

    Returns:
        Tuple[LDAModel, DataFrame]: [description]

        output_df
        root
        |-- product_name: string (nullable = true)
        |-- brands: array (nullable = true)
        |    |-- element: string (containsNull = true)
        |-- categories: array (nullable = true)
        |    |-- element: string (containsNull = true)
        |-- topicDistribution: vector (nullable = true)
        |-- score: float (nullable = true)
        |-- topic: integer (nullable = true)
    """
    lda = LDA(k=k, maxIter=iter, seed=seed, featuresCol=col)
    model = lda.fit(df)
    output_df = model.transform(df).withColumn(
        'score_topic', udf.get_topic(F.col('topicDistribution'))
    ).select('product_name', 'brands', 'categories', 'topicDistribution',
             F.col('score_topic.score').alias('score'),
             F.col('score_topic.topic').alias('topic'))

    return model, output_df


def to_model_topics(model: LocalLDAModel, vocab: list, topics_num: int = 10) -> DataFrame:
    """[summary]

    Args:
        model (LocalLDAModel): [description]
        vocab (list): [description]
        topics_num (int, optional): [description]. Defaults to 10.

    Returns:
        DataFrame: [description]
    """
    topics = model.describeTopics(topics_num)
    total_vocabulary = ','.join(vocab)
    return topics.withColumn("words", udf.term_to_Word(topics.termIndices, F.lit(total_vocabulary))
                             ).withColumn('top_terms', F.arrays_zip('words', 'termWeights')).select('topic', 'top_terms')


def to_lda_vis_stats(tf_idf_df: DataFrame, item_topic_df: DataFrame, vocabulary: list, model: LDAModel) -> dict:
    """
    generate data set for ladvis visualization

    Args:
        tf_idf_df (DataFrame): [description]
        item_topic_df (DataFrame): [description]
        vocabulary (list): [description]
        model (LDAModel): [description]

    Returns:
        dict: [description]
    """

    def _filter(data):
        topic_term_dists = []
        doc_topic_dists = []

        for x, y in zip(data['doc_topic_dists'], data['doc_lengths']):
            if np.sum(x) == 1:
                topic_term_dists.append(x)
                doc_topic_dists.append(y)

        data['doc_topic_dists'] = topic_term_dists
        data['doc_lengths'] = doc_topic_dists
        return data

    word_counts_agg = tf_idf_df.select((F.explode('cleaned_words')).alias(
        "words")).groupby("words").count()
    word_counts_dict = {r['words']: r['count']
                        for r in word_counts_agg.collect()}

    visual_stats = {
        'doc_lengths': [r[0] for r in tf_idf_df.select(F.size('cleaned_words')).collect()],
        'vocab': vocabulary,
        'term_frequency': [word_counts_dict[w] for w in vocabulary],
        'topic_term_dists': np.array(model.topicsMatrix().toArray()).T,
        'doc_topic_dists': np.array(
            [
                x['topicDistribution'].toArray()
                for x in item_topic_df.select("topicDistribution").collect()
            ])
    }

    return _filter(visual_stats)
