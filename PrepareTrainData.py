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
import pandas as pd
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from scipy.stats import chi2_contingency

conf = SparkConf().setAppName("prepare test data") \
    .setMaster("local[7]").set('spark.executors.memory', '16g') \
    .set('spark.driver.memory', '4g').set("spark.sql.execution.arrow.enabled", "true").set('spark.local.dir', './')
sc = SparkContext(conf=conf)
sql_context = SQLContext(sc)


def get_train_data():
    '''
    合并训练数据集
    '''
    train_user = ks.read_csv("../data/train_preliminary/user.csv")

    train_click_log = ks.read_csv("../data/train_preliminary/click_log.csv")

    train_ad = ks.read_csv("../data/train_preliminary/ad.csv")
    train_data = train_user.merge(train_click_log, on="user_id",
                                  how='inner').merge(train_ad,
                                                     on="creative_id",
                                                     how='inner')
    return train_data


def get_test_data():
    '''
    合并测试集
    '''
    test_click_log = ks.read_csv("../data/test/click_log.csv")
    test_ad = ks.read_csv("../data/test/ad.csv")
    test_data = test_click_log.merge(test_ad, on="creative_id")
    return test_data


def gen_embedding_corpus():
    '''
    和并训练和测试语料
    '''
    train_data = get_train_data()
    train_data.to_csv('./data/train_corpus.csv', index=False, num_files=1, header=True)


def train_data_analysis(train_data):
    # train_data.head()
    # Row(creative_id='100010',
    #     user_id='668133',
    #     age='5',
    #     gender='2',
    #     time='19',
    #     click_times='1',
    #     ad_id='91063',
    #     product_id='\\N',
    #     product_category='18',
    #     advertiser_id='34653',
    #     industry='302')

    test_result = []
    for col_name in train_data.columns:
        if 'age' != col_name:
            relation_df = train_data.groupBy(col_name,
                                             'age').count().toPandas()
            freq_table = pd.pivot_table(relation_df,
                                        index=[col_name],
                                        columns=["age"],
                                        values=["count"])
            freq_table.fillna(0, inplace=True)
            kf_value = chi2_contingency(freq_table.values)
            print(f'{col_name} chi2 test pvalue {kf_value[1]}')
            test_result.append([col_name, kf_value[1]])
    pd.DataFrame(test_result, columns=['col_name',
                                       'pvalue']).to_csv("chi2_test.csv",
                                                         index=False)


def get_train_corpus():
    train_data = get_train_data()
    query = '''
    select user_id,
           age,
           gender,
           concat_ws(' ', collect_list(b.time))             time1,
           concat_ws(' ', collect_list(b.creative_id))      creative_id,
           concat_ws(' ', collect_list(b.click_times))      click_times,
           concat_ws(' ', collect_list(b.ad_id))            ad_id,
           concat_ws(' ', collect_list(b.product_id))       product_id,
           concat_ws(' ', collect_list(b.product_category)) product_category,
           concat_ws(' ', collect_list(b.advertiser_id))    advertiser_id,
           concat_ws(' ', collect_list(b.industry))         industry
    from {train_data} b
    group by user_id,
             age,
             gender
    order by time1 asc
    '''
    train_encode_result = ks.sql(query=query, train_data=train_data)
    train_encode_result.to_csv('../data/train_test_corpus', index=False, num_files=1)


def get_test_corpus():
    test_data = get_test_data()
    query = '''
    select user_id,
           concat_ws(' ', collect_list(b.time))             time1,
           concat_ws(' ', collect_list(b.creative_id))      creative_id,
           concat_ws(' ', collect_list(b.click_times))      click_times,
           concat_ws(' ', collect_list(b.ad_id))            ad_id,
           concat_ws(' ', collect_list(b.product_id))       product_id,
           concat_ws(' ', collect_list(b.product_category)) product_category,
           concat_ws(' ', collect_list(b.advertiser_id))    advertiser_id,
           concat_ws(' ', collect_list(b.industry))         industry
    from {test_data} b
    group by user_id
    order by time1 asc
    '''

    test_encode_result = ks.sql(query=query, test_data=test_data)
    test_encode_result.to_csv('../data/predict_corpus', index=False, num_files=1)


def get_ad_dict():
    train_ad = ks.read_csv("../data/train_preliminary/ad.csv")
    test_ad = ks.read_csv("../data/test/ad.csv")
    ad_info = ks.concat([train_ad, test_ad], axis=0)
    ad_info = ad_info.drop_duplicates()
    ad_dict_sql = '''
     select 
       creative_id,
       product_id,
       product_category,
       advertiser_id,
       industry,
       row_number()
       over (partition by product_id, product_category,advertiser_id,industry order by 1 desc) ad_rn
       from {ad_info}
    '''
    ad_info = ks.sql(ad_dict_sql, ad_info=ad_info)
    print(ad_info.nunique())
    ad_info.to_csv('../data/ad_info', index=False, num_files=1)


# creative_id         3412772
# product_id            39057
# product_category         18
# advertiser_id         57870
# industry                332
# ad_rn                 14795

if __name__ == "__main__":
    # get_train_corpus()
    get_test_corpus()