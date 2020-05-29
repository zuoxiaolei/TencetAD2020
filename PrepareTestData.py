'''
__project_ = 'tencentAD2020'
__author__ = zuoxiaolei
__time__ = '2020/5/24 20:44'
__description = ''
'''
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

conf = SparkConf().setAppName("prepare test data") \
    .setMaster("local[*]")
sc = SparkContext(conf=conf)
sql_context = SQLContext(sc)

test_click = sql_context.read.csv("../data/test/click_log.csv", header=True)
print(test_click.head())

test_click.createOrReplaceTempView('test_click')
word2vec_corpus_sql = '''
    select user_id,
            collect_set(creative_id) sentence
    from test_click
    group by user_id
'''
test_click_group = sql_context.sql(word2vec_corpus_sql).toPandas()
test_click_group.loc[:, 'sentence'] = test_click_group.loc[:, 'sentence'].map(lambda x: [int(ele) for ele in x])
test_click_group.to_csv('testClick.csv', index=False)
