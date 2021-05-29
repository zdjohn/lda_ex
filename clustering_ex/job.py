from clustering_ex import utils, transform
import time


def run(session, settings):
    """ load data, transfrom, save

    Args:
        session ([type]): spark session
        settings ([type]): [description]
    """

    # args init
    k = int(settings.get('k', 5))
    iteration = settings.get('iteration', 5)
    versioned_output = f"output/{settings.get('version', f'{int(time.time())}')}_{k}"
    random_seed = int(settings.get('seed', 0))

    # extract
    raw_df = session.read.json(settings['source_file'])

    # transfrom
    item_df = transform.to_item_df(raw_df)
    vocab, tf_idf_feature_df = transform.to_tf_idf_features_df(item_df)
    model, item_topic_df = transform.fit_lda(
        tf_idf_feature_df, k, iteration, random_seed, 'features')
    topics_df = transform.to_model_topics(model, vocab, 10)

    # load data to s3
    item_df.write.mode('overwrite').parquet(
        f'{settings["base_bucket"]}/{versioned_output}/item_df.parquet')
    tf_idf_feature_df.write.mode('overwrite').parquet(
        f'{settings["base_bucket"]}/{versioned_output}/item_tf_idf_df.parquet')
    item_topic_df.write.mode('overwrite').parquet(
        f'{settings["base_bucket"]}/{versioned_output}/item_topic_df.parquet')
    topics_df.write.mode('overwrite').parquet(
        f'{settings["base_bucket"]}/{versioned_output}/topics_df.parquet')

    # save model artifacts
    utils.save_model_s3(
        model, settings['base_bucket'], f'{versioned_output}/lda_model')

    # upload visual stats
    lda_vis_stats = transform.to_lda_vis_stats(
        tf_idf_feature_df, item_topic_df, vocab, model)
    utils.upload_lda_vis_data_s3(
        lda_vis_stats, settings['base_bucket'], f'{versioned_output}/lda_vis')
