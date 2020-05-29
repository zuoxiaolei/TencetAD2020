import logging
from datetime import datetime

import gensim
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
conf = SparkConf().setAppName("ad_classify").setMaster("local[*]")
sc = SparkContext(conf=conf)
sql_context = SQLContext(sc)

word2vec_corpus_sql = '''
    select concat_ws(' ', collect_set(creative_id)) sentence
    from click_log
    group by user_id
'''


# def group_concat(values):
#     return values
# sql_context.udf.register("group_concat", group_concat)


def get_data(filename='data/train_preliminary/click_log.csv'):
    df_train = sql_context.read.format('com.databricks.spark.csv') \
        .options(header='true', inferschema='true').load(
        filename)
    return df_train


def prepare_data():
    start_time = datetime.now()
    train_df = get_data()
    test_df = get_data(filename='data/test/click_log.csv')
    user_click_log = train_df.unionAll(test_df)
    user_click_log.createTempView("click_log")

    corpus = sql_context.sql(word2vec_corpus_sql)
    corpus.summary().show()
    corpus.coalesce(1).write.csv("corpus.csv")
    print(datetime.now() - start_time)


def train_word2vec():
    model = gensim.models.Word2Vec(corpus_file='corpus.csv/part-00000-8e121987-6686-4d66-8888-3af880559b03-c000.csv',
                                   window=5, min_count=1, size=50)
    model.wv.save_word2vec_format("data/word2vec.bin", binary=True)


def get_embedding():
    embedding_size = 50
    from NNAD import prepare
    import numpy as np
    model = gensim.models.KeyedVectors.load_word2vec_format("data/word2vec.bin", binary=True)
    a, w2v_corpus = prepare()
    vocab_dict = {ele: num + 1 for num, ele in enumerate(a)}
    index_vocab_rel = dict(zip(vocab_dict.values(), vocab_dict.keys()))
    embeddings = [np.random.randn(embedding_size)]

    for ele in range(1, len(a) + 1):
        vocab = index_vocab_rel[ele]
        embeddings.append(model[str(vocab)])
    embeddings.append(np.random.randn(embedding_size))
    embeddings = np.array(embeddings)
    print(embeddings.shape)
    np.savez_compressed('embeddings.npz', embeddings)


if __name__ == '__main__':
    # model = gensim.models.KeyedVectors.load_word2vec_format("data/word2vec.model")
    # model = gensim.models.KeyedVectors.load_word2vec_format("data/word2vec.bin", binary=True)
    # model.wv.save_word2vec_format("data/word2vec.bin", binary=True)
    # print(model.most_similar('876'))
    # prepare_data()
    get_embedding()
    np.load