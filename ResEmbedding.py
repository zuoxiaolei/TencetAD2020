'''
__project_ = 'tencentAD2020'
__author__ = zuoxiaolei
__time__ = '2020/5/27 22:12'
__description = ''
'''
import csv
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from tensorflow import keras
from tqdm import tqdm
csv.field_size_limit(500 * 1024 * 1024)

def get_ad_dict():
    '''
    获取广告ID重新编码
    :return:
    '''
    ad_dict_df = pd.read_csv('data/ad_info/part-00000-aae701d2-8502-4984-8e5b-59bdac6c9932-c000.csv')
    ad_dict_df = ad_dict_df.loc[:, ['creative_id', 'ad_rn']]
    ad_dict_df.loc[:, 'creative_id'] = ad_dict_df['creative_id'].apply(str)
    ad_dict_df.loc[:, 'ad_rn'] = ad_dict_df['ad_rn'].apply(int)
    return dict(ad_dict_df.values.tolist())


def train_test_split():
    train_data_path = './data/train.csv'
    test_data_path = "./data/test.csv"
    n = 0
    with open(train_data_path, 'w', encoding='utf8') as fh_train:
        with open(test_data_path, 'w', encoding='utf8') as fh_test:
            with open('data/train_test_corpus/part-00000-92cc7051-a5ec-4f2a-b5cf-9931e1b6e00f-c000.csv', 'r',
                      encoding='utf8') as fh_train_test:
                for line in fh_train_test:
                    n = n + 1
                    if n > 720000:
                        fh_test.write(line)
                    else:
                        fh_train.write(line)


ad_dict = get_ad_dict()


class RESEmbedding(keras.Model):

    def __init__(self, embedding_dim=200, mask_zero=True,
                 hidden_units=768, class_num=10):
        super(RESEmbedding, self).__init__()

        self.time_embedding = keras.layers.Embedding(91 + 2, embedding_dim, mask_zero=mask_zero)
        self.click_times_embedding = keras.layers.Embedding(94 + 2, embedding_dim, mask_zero=mask_zero)
        self.product_id_embedding = keras.layers.Embedding(50057 + 2, embedding_dim, mask_zero=mask_zero)
        self.product_category_embedding = keras.layers.Embedding(18 + 2, embedding_dim, mask_zero=mask_zero)
        self.advertiser_id_embedding = keras.layers.Embedding(67870 + 2, embedding_dim, mask_zero=mask_zero)
        self.industry_embedding = keras.layers.Embedding(432 + 2, embedding_dim, mask_zero=mask_zero)
        self.creative_id_embedding = keras.layers.Embedding(14795 + 2, embedding_dim, mask_zero=mask_zero)

        self.time_weight = tf.Variable(random.random(), trainable=True)
        self.click_times = tf.Variable(random.random(), trainable=True)
        self.product_id = tf.Variable(random.random(), trainable=True)
        self.product_category = tf.Variable(random.random(), trainable=True)
        self.advertiser_id = tf.Variable(random.random(), trainable=True)
        self.industry = tf.Variable(random.random(), trainable=True)
        self.creative_id = tf.Variable(random.random(), trainable=True)

        self.lstm = keras.layers.LSTM(units=hidden_units, return_sequences=False)
        self.dense = keras.layers.Dense(units=class_num, activation='softmax')

    def call(self, input_tensors):
        time_feature = self.time_embedding(input_tensors[0]) * self.time_weight
        click_times_feature = self.click_times_embedding(input_tensors[1]) * self.click_times
        product_id_feature = self.product_id_embedding(input_tensors[2]) * self.product_id
        product_category_feature = self.product_category_embedding(input_tensors[3]) * self.product_category
        advertiser_id_feature = self.advertiser_id_embedding(input_tensors[4]) * self.advertiser_id
        industry_feature = self.industry_embedding(input_tensors[5]) * self.industry
        creative_id_feature = self.creative_id_embedding(input_tensors[6]) * self.creative_id
        feature_add = time_feature + click_times_feature + product_id_feature \
                      + product_category_feature + advertiser_id_feature + industry_feature + creative_id_feature
        lstm_feature = self.lstm(feature_add)
        out = self.dense(lstm_feature)
        return out


class DataGenerator(keras.utils.Sequence):
    def __init__(self, filename, batch_size=256, user_num=720000, is_age=True):
        self.batch_size = batch_size
        self.max_len = 30
        self.filename = filename
        self.user_num = user_num
        self.generate = self.get_batch_sample()
        self.is_age = is_age

    def __getitem__(self, index):
        x, y = next(self.generate)
        return x, y

    def get_feature_index(self, string):
        return [int(ele) if ele != '\\N' else 0 for ele in string.split(' ')]

    def get_sample(self, filename):
        while 1:
            with open(filename, 'r', encoding='utf-8') as fh:
                lines = csv.DictReader(fh)
                for line in lines:
                    time = self.get_feature_index(line['time1'])
                    click_times = self.get_feature_index(line['click_times'])
                    creative_id = [ad_dict[ele] for ele in line['creative_id'].split(' ')]
                    product_id = self.get_feature_index(line['product_id'])
                    product_category = self.get_feature_index(line['product_category'])
                    advertiser_id = self.get_feature_index(line['advertiser_id'])
                    industry = self.get_feature_index(line['industry'])
                    age = int(line["age"]) - 1
                    gender = int(line["gender"]) - 1
                    yield time, click_times, product_id, product_category, advertiser_id, industry, creative_id, age, gender

    def get_batch_sample(self):
        times = []
        click_times_list = []
        product_ids = []
        product_categorys = []
        advertiser_ids = []
        industrys = []
        creative_ids = []
        ages = []
        genders = []
        for time, click_times, product_id, \
            product_category, advertiser_id, \
            industry, creative_id, age, gender in self.get_sample(filename=self.filename):
            times.append(time)
            click_times_list.append([ele if ele < 96 else 0 for ele in click_times])
            product_ids.append(product_id)
            product_categorys.append(product_category)
            advertiser_ids.append(advertiser_id)
            industrys.append(industry)
            creative_ids.append(creative_id)
            if self.is_age:
                ages.append(age)
            else:
                ages.append(gender)
            if len(times) >= self.batch_size:
                maxlen = min([max([len(ele) for ele in times]), 50])
                yield np.array([pad_sequences(times, maxlen=maxlen),
                                pad_sequences(click_times_list, maxlen=maxlen),
                                pad_sequences(product_ids, maxlen=maxlen),
                                pad_sequences(product_categorys, maxlen=maxlen),
                                pad_sequences(advertiser_ids, maxlen=maxlen),
                                pad_sequences(industrys, maxlen=maxlen),
                                pad_sequences(creative_ids, maxlen=maxlen),
                                ]), np.array(ages)
                times = []
                click_times_list = []
                product_ids = []
                product_categorys = []
                advertiser_ids = []
                industrys = []
                creative_ids = []
                ages = []
                genders = []

    def __len__(self):
        return int(self.user_num / self.batch_size)


def train_model(model_name='gender', is_age=False):
    class_num = 10 if is_age else 2
    model = RESEmbedding(class_num=class_num)
    model.build(input_shape=(8, None, None))
    model.summary()
    model.compile(optimizer='adam',
                  metrics=[keras.metrics.sparse_categorical_accuracy],
                  loss=keras.losses.sparse_categorical_crossentropy)

    train_generate = DataGenerator(
        filename='data/train_test_corpus/part-00000-92cc7051-a5ec-4f2a-b5cf-9931e1b6e00f-c000.csv',
        user_num=900000,
        is_age=is_age)
    test_generate = DataGenerator(filename='./data/test.csv',
                                  user_num=180000, is_age=is_age)
    model.fit(x=train_generate,
              validation_data=test_generate,
              callbacks=[keras.callbacks.EarlyStopping(patience=2),
                         keras.callbacks.ModelCheckpoint('{}.h5'.format(model_name))],
              epochs=1)
    model.save_weights('final_{}.h5'.format(model_name))


def get_feature_index(string):
    return [int(ele) if ele != '\\N' else 0 for ele in string.split(' ')]


def get_predict_corpus(batch_size=1024):
    '''
    :return:
    '''
    times = []
    click_times_list = []
    product_ids = []
    product_categorys = []
    advertiser_ids = []
    industrys = []
    creative_ids = []
    user_ids = []

    with open('data/predict_corpus/part-00000-34fdfae6-514a-474c-8254-baedcc417c79-c000.csv', 'r', encoding='utf-8') as fh:
        lines = csv.DictReader(fh)
        for line in lines:
            time = get_feature_index(line['time1'])
            click_times = get_feature_index(line['click_times'])
            creative_id = [ad_dict[ele] for ele in line['creative_id'].split(' ')]
            product_id = get_feature_index(line['product_id'])
            product_category = get_feature_index(line['product_category'])
            advertiser_id = get_feature_index(line['advertiser_id'])
            industry = get_feature_index(line['industry'])

            user_ids.append(line['user_id'])
            times.append(time)
            click_times_list.append([ele if ele < 96 else 0 for ele in click_times])
            product_ids.append(product_id)
            product_categorys.append(product_category)
            advertiser_ids.append(advertiser_id)
            industrys.append(industry)
            creative_ids.append(creative_id)
            if len(times) >= batch_size:
                maxlen = min([max([len(ele) for ele in times]), 50])
                yield user_ids, np.array([pad_sequences(times, maxlen=maxlen),
                                pad_sequences(click_times_list, maxlen=maxlen),
                                pad_sequences(product_ids, maxlen=maxlen),
                                pad_sequences(product_categorys, maxlen=maxlen),
                                pad_sequences(advertiser_ids, maxlen=maxlen),
                                pad_sequences(industrys, maxlen=maxlen),
                                pad_sequences(creative_ids, maxlen=maxlen),
                                ])
                times = []
                click_times_list = []
                product_ids = []
                product_categorys = []
                advertiser_ids = []
                industrys = []
                creative_ids = []
                user_ids = []
        if times:
            yield user_ids, np.array([pad_sequences(times, maxlen=maxlen),
                                      pad_sequences(click_times_list, maxlen=maxlen),
                                      pad_sequences(product_ids, maxlen=maxlen),
                                      pad_sequences(product_categorys, maxlen=maxlen),
                                      pad_sequences(advertiser_ids, maxlen=maxlen),
                                      pad_sequences(industrys, maxlen=maxlen),
                                      pad_sequences(creative_ids, maxlen=maxlen),
                                      ])


def predict_model():
    user_ids = []
    predicted_age = []
    predicated_gender = []
    model = RESEmbedding(class_num=10)
    model.build(input_shape=(8, None, None))
    model.load_weights('final_age.h5')
    for user, feature in tqdm(get_predict_corpus()):
        user_ids.extend(user)
        predicted_age.extend((np.argmax(model(feature), axis=1)+1).tolist())
    del model
    keras.backend.clear_session()

    model = RESEmbedding(class_num=2)
    model.build(input_shape=(8, None, None))
    model.load_weights('final_gender.h5')
    for user, feature in tqdm(get_predict_corpus()):
        predicated_gender.extend((np.argmax(model(feature), axis=1)+1).tolist())

    result = pd.DataFrame(list(zip(user_ids, predicted_age, predicated_gender)),
                          columns=['user_id', 'predicted_age', 'predicted_gender'])
    result.to_csv('result.csv', index=False)



if __name__ == '__main__':
    train_model()
    train_model(model_name='age', is_age=True)
    predict_model()