from itertools import chain
import pandas as pd
from pyspark.sql.functions import udf, pandas_udf
from pyspark.sql.types import ArrayType, StringType, IntegerType


@pandas_udf(ArrayType(StringType()))
def merge_lists(list_col_df: pd.Series) -> pd.Series:
    res = []
    for row in list_col_df:
        res.append(list(set(chain(*row))))
    return pd.Series(res)


@pandas_udf(ArrayType(StringType()))
def term_to_Word(term_idx_col_df: pd.Series, vocabulary: pd.Series) -> pd.Series:
    vocab = vocabulary.iloc[0].split(',')
    res = []
    for row in term_idx_col_df:
        words = []
        for termID in row:
            words.append(vocab[termID])
        res.append(words)
    return pd.Series(res)


@udf(IntegerType())
def get_topic(x):
    return x.tolist().index(max(x))
