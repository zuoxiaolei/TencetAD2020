#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   PrepareTrainData.py
@Time    :   2020/05/25 12:33:02
@Author  :   zuoxiaolei 
@Desc    :   None
'''
import os

os.environ['ARROW_PRE_0_15_IPC_FORMAT'] = '1'
import databricks.koalas as ks
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

conf = SparkConf().setAppName("prepare test data") \
    .setMaster("spark://47.105.217.91:7077").set('spark.executors.memory', '16g') \
    .set('spark.driver.memory', '10g').set("spark.sql.execution.arrow.enabled", "true").set(
    "spark.debug.maxToStringFields", "100").set('spark.executor.memory', '2g')
sc = SparkContext(conf=conf)
sql_context = SQLContext(sc)


def get_train_data():
    '''
    合并训练数据集
    '''
    train_click_log = ks.read_csv("data/train_preliminary/click_log.csv")
    train_ad = ks.read_csv("./data/train_preliminary/ad.csv")
    train_data = train_click_log.merge(train_ad, on="creative_id", how='inner')
    test_click_log = ks.read_csv("./data/test/click_log.csv")
    test_ad = ks.read_csv("./data/test/ad.csv")
    test_data = test_click_log.merge(test_ad, on="creative_id")
    user_data = ks.concat([train_data, test_data], axis=0)
    # user_data.cache()
    sql = '''
        SELECT user_id,
           count(distinct a.time)                                                                 time_count,
           count(a.time) / if(max(a.time) = min(a.time), 0.1, max(a.time) - min(a.time)) time_intervel, -- 点击广告间隔
           count(DISTINCT creative_id)                                                   creative_id_ncount,
           count(creative_id)                                                            creative_id_count,
           sum(click_times)                                                              click_times_sum,
           avg(click_times)                                                              click_times_avg,
           std(click_times)                                                              click_times_std,
           count(DISTINCT ad_id)                                                         ad_id_ncount,
           count(DISTINCT product_id)                                                    product_id_ncount,
           count(DISTINCT product_category)                                              product_category_ncount,
           count(DISTINCT advertiser_id)                                                 advertiser_id_ncount,
           count(DISTINCT industry)                                                      industry_ncount
    FROM {user_data} a
    GROUP BY user_id
    '''
    stats_features = ks.sql(sql, user_data=user_data)
    print(stats_features.shape)
    stats_features.to_csv('./data/stats_features', num_files=1)


if __name__ == '__main__':
    get_train_data()
