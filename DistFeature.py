'''
__project_ = 'tencentAD2020'
__author__ = zuoxiaolei
__time__ = '2020/5/30 22:05'
__description = ''
'''
import databricks.koalas as ks
from pyspark import SparkConf, SparkContext
import pandas as pd

conf = SparkConf().setMaster("local[7]").setAppName('ad eda')
sc = SparkContext(conf=conf)


def get_age_feature():
    train_user = ks.read_csv("./data/user_train.csv")
    train_click_log = ks.read_csv("data/train_preliminary/click_log.csv")
    train_data = train_user.merge(train_click_log, on="user_id", how='inner')
    sql = '''
    select creative_id,
            age,
            sum(nvl(click_times, 0)) click_times
    from {train_data}
    group by  creative_id, age
    '''
    age_data = ks.sql(sql, train_data=train_data)
    age_data.cache()
    sql = '''
    SELECT creative_id,
           age,
           click_times / sum(click_times)
                             OVER (PARTITION BY creative_id  ORDER BY click_times DESC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) age_dist
    FROM {age_data}
    '''
    age_dist_data = ks.sql(sql, age_data=age_data)
    age_dist_data.head(10)
    age_dist_data.cache()
    age_dist_pivot = age_dist_data.pivot(index='creative_id',
                                         columns='age',
                                         values='age_dist')
    age_dist_pivot.columns = ['age_' + str(ele) for ele in range(1, 11)]
    age_dist_pivot = age_dist_pivot.reset_index()
    age_dist_pivot.fillna(0, inplace=True)
    age_dist_pivot.to_csv('./data/age_dist', num_files=1)


def get_gender_feature():
    train_user = ks.read_csv("data/train_preliminary/user.csv")
    train_click_log = ks.read_csv("data/train_preliminary/click_log.csv")
    train_data = train_user.merge(train_click_log, on="user_id", how='inner')
    sql = '''
    select creative_id,
            gender,
            sum(nvl(click_times, 0)) click_times
    from {train_data}
    group by  creative_id, gender
    '''
    age_data = ks.sql(sql, train_data=train_data)
    age_data.cache()
    sql = '''
    SELECT creative_id,
           gender,
           click_times / sum(click_times)
                             OVER (PARTITION BY creative_id  ORDER BY click_times DESC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) gender_dist
    FROM {age_data}
    '''
    age_dist_data = ks.sql(sql, age_data=age_data)
    age_dist_data.head(10)
    age_dist_data.cache()
    age_dist_pivot = age_dist_data.pivot(index='creative_id',
                                         columns='gender',
                                         values='gender_dist')
    age_dist_pivot.columns = ['gender_' + str(ele) for ele in range(1, 3)]
    age_dist_pivot = age_dist_pivot.reset_index()
    age_dist_pivot.fillna(0, inplace=True)
    age_dist_pivot.to_csv('./data/gender_dist', num_files=1)


def train_test_split(split_size=0.2):
    user = pd.read_csv('data/train_preliminary/user.csv')
    user = user.sample(frac=1)
    user_train = user.iloc[0:int(user.shape[0]*(1-split_size))]
    user_test = user.iloc[int(user.shape[0] * (1 - split_size)):]
    user_train.to_csv("./data/user_train.csv", index=False)
    user_test.to_csv("./data/user_test.csv", index=False)


if __name__ == '__main__':
    # train_test_split()
    get_age_feature()
    # get_gender_feature()