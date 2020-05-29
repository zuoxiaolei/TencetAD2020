import gensim
import pandas as pd
from gensim.similarities.index import AnnoyIndexer
from tqdm import tqdm
from sklearn.metrics import classification_report
import logging
from tqdm import tqdm
import numpy as np
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pyspark.sql import SQLContext
from pyspark import SparkConf, SparkContext
from datetime import datetime
import keras
from main import PopulationModel
conf = SparkConf().setAppName("ad_classify").setMaster("local[*]")
sc = SparkContext(conf=conf)
sql_context = SQLContext(sc)

word2vec_corpus_sql = '''
    select user_id,
            concat_ws(' ', collect_set(concat_ws('|', creative_id, click_times))) sentence
    from click_log
    group by user_id
'''

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
    corpus.coalesce(1).write.csv("train_date")
    print(datetime.now() - start_time)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def evaluate():
    user_info = pd.read_csv('data/train_preliminary/user.csv')
    user_sample = user_info.sample(frac=0.2)
    model = gensim.models.KeyedVectors.load_word2vec_format("data/word2vec.bin", binary=True)
    print(model[['1', '10']])
    print(model[['1']])
    print(model[['10']])
    # annoy_index = AnnoyIndexer(model, 5)
    # res = []
    # for ele in tqdm(user_sample['user_id']):
    #     select = 1
    #     for ele in model.most_similar(str(ele), topn=10, indexer=annoy_index)[1:]:
    #         ele = ele[0]
    #         if int(ele) <= 900000:
    #             select = int(ele)
    #     res.append([select])
    # res = pd.DataFrame(res, columns=['user_id'])
    # res = res.merge(user_info, on=["user_id"])
    # print(classification_report(user_sample['age'].tolist(), res['age'].tolist()))
    # print(classification_report(user_sample['gender'].tolist(), res['gender'].tolist()))


def evaluate1():
    model = gensim.models.KeyedVectors.load_word2vec_format("data/word2vec.bin", binary=True)
    user = pd.read_csv('./data/train_preliminary/user.csv')
    train_corpus = pd.read_csv('train_date/part-00000-4bae5739-ee39-4297-bfbf-6b2ed4a0e3f1-c000.csv', header=None)
    train_corpus.columns = ['user_id', 'sentence']
    user = user.merge(train_corpus, on="user_id")
    y1 = user.gender-1
    y2 = user.age-1
    features = []
    word2vec_feature = []
    _, share_model = PopulationModel().build_simple_model()
    share_model.load_weights('data/share_layer.h5')
    tmp_features = []
    for ele in tqdm(user.sentence.tolist()):
        words = ele.split(' ')
        tmp = model.wv[words].mean(axis=0)
        tmp_features.append(tmp)
        word2vec_feature.append(tmp)
        if len(tmp_features) >= 256:
            triple_feature = share_model.predict(np.array(tmp_features), batch_size=len(tmp_features))
            features.append(triple_feature)
            tmp_features = []
    if tmp_features:
        triple_feature = share_model.predict(np.array(tmp_features), batch_size=len(tmp_features))
        features.append(triple_feature)
    features = np.concatenate(features, axis=0)
    features = np.concatenate([np.array(word2vec_feature), features], axis=1)
    print(features.shape)
    X_train, X_test, y_train, y_test = train_test_split(features, y1)
    lightgbm = LGBMClassifier(n_estimators=200)
    lightgbm.fit(X_train, y_train,  eval_set=[(X_test, y_test)],
                 early_stopping_rounds=5)
    pred = lightgbm.predict(X_test)
    print(classification_report(y_test, pred))

    X_train, X_test, y_train, y_test = train_test_split(features, y2)
    lightgbm = LGBMClassifier(n_estimators=200)
    lightgbm.fit(X_train, y_train,  eval_set=[(X_test, y_test)],
                 early_stopping_rounds=5)
    pred = lightgbm.predict(X_test)
    print(classification_report(y_test, pred))


if __name__ == '__main__':
    # prepare_data()
    evaluate1()