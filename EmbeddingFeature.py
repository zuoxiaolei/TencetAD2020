import logging
from datetime import datetime

import gensim
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
conf = SparkConf().setAppName("ad_classify").setMaster("local[*]")
sc = SparkContext(conf=conf)
sql_context = SQLContext(sc)


def get_data(filename='data/train_preliminary/click_log.csv'):
    df_train = sql_context.read.format('com.databricks.spark.csv').options(
        header='true', inferschema='true').load(filename)
    return df_train


def prepare_data():
    start_time = datetime.now()
    train_df = get_data()
    test_df = get_data(filename='data/test/click_log.csv')
    user_click_log = train_df.unionAll(test_df)
    user_click_log.createTempView("click_log")
    word2vec_corpus_sql = '''
        select concat_ws(' ', collect_list(creative_id)) sentence
        from click_log
        group by user_id
    '''
    corpus = sql_context.sql(word2vec_corpus_sql)
    corpus.summary().show()
    corpus.coalesce(1).write.csv("corpus")
    print(datetime.now() - start_time)


def train_word2vec(size=100):
    model = gensim.models.Word2Vec(
        corpus_file=
        'corpus/part-00000-8274d92c-217e-4ce7-80c7-50c52a899545-c000.csv',
        window=5,
        min_count=1,
        size=size)
    model.wv.save_word2vec_format("data/word2vec.bin", binary=True)


if __name__ == '__main__':
    # prepare_data()
    train_word2vec()
