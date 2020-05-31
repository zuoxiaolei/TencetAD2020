import logging
from datetime import datetime

import databricks.koalas as ks
import gensim
import joblib
import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
conf = SparkConf().setAppName("evaluate tencent ad").setMaster("spark://47.105.217.91:7077").set('spark.driver.memory',
                                                                                                 '10g').set(
    'spark.driver.memory', '10g').set("spark.sql.execution.arrow.enabled", "true").set("spark.debug.maxToStringFields",
                                                                                       "100").set(
    'spark.executor.memory', '2g')
sc = SparkContext(conf=conf)
sql_context = SQLContext(sc)

word2vec_corpus_sql = '''
    select user_id,
            concat_ws(' ', collect_list(creative_id)) sentence
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


def get_wv_features():
    model = gensim.models.KeyedVectors.load_word2vec_format("data/word2vec.bin", binary=True)
    train_corpus = pd.read_csv('train_date/part-00000-9134d416-d4f5-4998-96ce-8eeed9133a94-c000.csv', header=None)
    train_corpus.columns = ['user_id', 'sentence']
    user = train_corpus
    word2vec_features = []

    user_ids = user["user_id"].values
    for ele in tqdm(user.sentence.tolist()):
        words = ele.split(' ')
        tmp = model.wv[words].mean(axis=0)
        word2vec_features.append(tmp)
    features = np.array(word2vec_features)
    features = pd.DataFrame(features)
    features.columns = ['wv' + str(ele + 1) for ele in range(100)]
    features.loc[:, "user_id"] = user_ids
    features.to_csv('./data/wv_features.csv', index=False)


def evaluate_age():
    features = pd.read_csv('data/combine_feature/part-00000-380aaa4b-c838-43f4-8cb7-80164a4256f2-c000.csv')
    y = features.age.values
    features.drop(['user_id', 'age', 'gender'], axis=1, inplace=True)
    print(features.shape)
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2)
    lightgbm = LGBMClassifier(n_estimators=200,
                              num_leaves=100,
                              feature_fraction=0.75,
                              bagging_fraction=0.75,
                              learning_rate=0.1
                              )
    lightgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                 early_stopping_rounds=5)
    pred = lightgbm.predict(X_test)
    print(classification_report(y_test, pred))
    joblib.dump(lightgbm, 'data/lgb_age')


def combine_feature(train=True):
    user_filename = 'data/train_preliminary/user.csv' if train else './data/test/click_log.csv'
    result_filename = './data/combine_feature' if train else './data/combine_feature_test'
    user_df = ks.read_csv(user_filename)
    if not train:
        user_df = ks.sql('select distinct user_id from {user_df}', user_df=user_df)
    wv_feature = ks.read_csv('data/wv_features.csv')
    nn_feature = ks.read_csv('data/nn_features.csv')
    stats_data = ks.read_csv("data/stats_features/part-00000-f6695da4-6d9f-4ba4-80b1-d370e636696b-c000.csv")
    all_features = user_df.merge(wv_feature, on='user_id').merge(nn_feature, on='user_id').merge(stats_data,
                                                                                                 on='user_id')
    print(all_features.shape)
    all_features.to_csv(result_filename, num_files=1)



if __name__ == '__main__':
    # prepare_data()
    # evaluate_age()
    combine_feature(train=False)
