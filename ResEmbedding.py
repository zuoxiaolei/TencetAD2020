'''
__project_ = 'tencentAD2020'
__author__ = zuoxiaolei
__time__ = '2020/5/27 22:12'
__description = ''
'''
import csv
from functools import partial
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from keras.utils.np_utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from tensorflow import keras

csv.field_size_limit(500 * 1024 * 1024)
logdir = 'logs/'
test_logdir = 'test_logs/'
train_writer = tf.summary.create_file_writer(logdir)


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


ad_dict = get_ad_dict()

loss_func = tf.keras.losses.categorical_crossentropy
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean()
train_metric = tf.keras.metrics.Accuracy(name='train_accuracy')
valid_loss = tf.keras.metrics.Mean()
valid_metric = tf.keras.metrics.Accuracy(name='valid_accuracy')


class RESEmbedding(keras.Model):

    def __init__(self, embedding_dim=100, mask_zero=True,
                 hidden_units=128, class_num=10):
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
        self.beta = tfp.distributions.beta.Beta(0.2, 0.2)

    def call(self, input_tensors, training=tf.constant(True)):
        time_feature = self.time_embedding(input_tensors[0]) * self.time_weight
        click_times_feature = self.click_times_embedding(input_tensors[1]) * self.click_times
        product_id_feature = self.product_id_embedding(input_tensors[2]) * self.product_id
        product_category_feature = self.product_category_embedding(input_tensors[3]) * self.product_category
        advertiser_id_feature = self.advertiser_id_embedding(input_tensors[4]) * self.advertiser_id
        industry_feature = self.industry_embedding(input_tensors[5]) * self.industry
        creative_id_feature = self.creative_id_embedding(input_tensors[6]) * self.creative_id
        feature_add = time_feature + click_times_feature + product_id_feature \
                      + product_category_feature + advertiser_id_feature + industry_feature + creative_id_feature
        if training:
            prob = self.beta.sample(1)
            index = tf.random.shuffle(tf.range(feature_add.shape[0]))
            feature_new = tf.gather(feature_add, index)
            embedding_feature = prob * feature_add + (1 - prob) * feature_new
            lstm_feature = self.lstm(embedding_feature)
            out = self.dense(lstm_feature)
            return out, index, prob

        else:
            prob = self.beta.sample(1)
            index = tf.random.shuffle(tf.range(feature_add.shape[0]))
            lstm_feature = self.lstm(feature_add)
            out = self.dense(lstm_feature)
            return out, index, prob


@tf.function
def train_step(model, feautes, labels):
    with tf.GradientTape() as tape:
        out, index, prob = model(feautes, training=tf.constant(True))
        mix_labels = tf.gather(labels, index) * prob + (1 - prob) * tf.gather(labels, index)
        loss = loss_func(mix_labels, out)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss.update_state(loss)
    train_metric.update_state(tf.math.argmax(labels, axis=-1), tf.math.argmax(out, axis=-1))


@tf.function
def valid_step(model, features, labels):
    predictions, index, prob = model(features, training=False)
    batch_loss = loss_func(labels, predictions)
    valid_loss.update_state(batch_loss)
    valid_metric.update_state(tf.math.argmax(labels, axis=-1), tf.math.argmax(predictions, axis=-1))


@tf.function
def train_model(model, ds_train, ds_valid, epochs, batch_size):
    for epoch in range(1, epochs + 1):
        step = tf.constant(1, dtype=tf.int64)
        for features, labels in ds_train:
            train_step(model, features, labels)
            with train_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=step)
                tf.summary.scalar('accurary', train_metric.result(), step=step)
            tf.print(tf.strings.format('Epoch {} Step {}/{} train_loss {} trainaccuracy {}', (epoch, step,
                                                                                              int(720000 / batch_size),
                                                                                              train_loss.result(),
                                                                                              train_metric.result())))
            step = step + 1

        valid_step_num = tf.constant(1, dtype=tf.int64)
        for features, labels in ds_valid:
            valid_step(model, features, labels)
            with train_writer.as_default():
                tf.summary.scalar('valid_loss', valid_loss.result(), step=valid_step_num)
                tf.summary.scalar('valid_accurary', valid_metric.result(), step=valid_step_num)
            valid_step_num = valid_step_num + 1
        tf.print(tf.strings.format(' valid_loss {} valid_accuracy {}', (valid_loss.result(),
                                                                        valid_metric.result())))
        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()


def get_feature_index(string):
    return [int(ele) if ele != '\\N' else 0 for ele in string.split(' ')]


def get_corpus_generate(filename='',
                        batch_size=64, training=True, is_age=True):
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
    labels = []

    class_num = 10 if is_age else 2

    with open(filename, 'r', encoding='utf-8') as fh:
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
            if training:
                if is_age:
                    labels.append(int(line['age']) - 1)
                else:
                    labels.append(int(line['gender']) - 1)

            if len(times) >= batch_size:
                maxlen = min([max([len(ele) for ele in times]), 50])
                if training:
                    yield np.array([pad_sequences(times, maxlen=maxlen),
                                    pad_sequences(click_times_list, maxlen=maxlen),
                                    pad_sequences(product_ids, maxlen=maxlen),
                                    pad_sequences(product_categorys, maxlen=maxlen),
                                    pad_sequences(advertiser_ids, maxlen=maxlen),
                                    pad_sequences(industrys, maxlen=maxlen),
                                    pad_sequences(creative_ids, maxlen=maxlen),
                                    ]), to_categorical(labels, num_classes=class_num)
                else:
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
                labels = []
        # if times:
        #     if training:
        #         yield np.array([pad_sequences(times, maxlen=maxlen),
        #                         pad_sequences(click_times_list, maxlen=maxlen),
        #                         pad_sequences(product_ids, maxlen=maxlen),
        #                         pad_sequences(product_categorys, maxlen=maxlen),
        #                         pad_sequences(advertiser_ids, maxlen=maxlen),
        #                         pad_sequences(industrys, maxlen=maxlen),
        #                         pad_sequences(creative_ids, maxlen=maxlen),
        #                         ]), to_categorical(labels, num_classes=class_num)
        #     else:
        #         yield user_ids, np.array([pad_sequences(times, maxlen=maxlen),
        #                                   pad_sequences(click_times_list, maxlen=maxlen),
        #                                   pad_sequences(product_ids, maxlen=maxlen),
        #                                   pad_sequences(product_categorys, maxlen=maxlen),
        #                                   pad_sequences(advertiser_ids, maxlen=maxlen),
        #                                   pad_sequences(industrys, maxlen=maxlen),
        #                                   pad_sequences(creative_ids, maxlen=maxlen),
        #                                   ])


# def predict_model():
#     user_ids = []
#     predicted_age = []
#     predicated_gender = []
#     model = RESEmbedding(class_num=10)
#     model.build(input_shape=(8, None, None))
#     model.load_weights('final_age.h5')
#     for user, feature in tqdm(get_predict_corpus()):
#         user_ids.extend(user)
#         predicted_age.extend((np.argmax(model(feature), axis=1) + 1).tolist())
#     del model
#     keras.backend.clear_session()
#
#     model = RESEmbedding(class_num=2)
#     model.build(input_shape=(8, None, None))
#     model.load_weights('final_gender.h5')
#     for user, feature in tqdm(get_predict_corpus()):
#         predicated_gender.extend((np.argmax(model(feature), axis=1) + 1).tolist())
#
#     result = pd.DataFrame(list(zip(user_ids, predicted_age, predicated_gender)),
#                           columns=['user_id', 'predicted_age', 'predicted_gender'])
#     result.to_csv('result.csv', index=False)


if __name__ == '__main__':
    class_num = 10
    batch_size = 128
    epoches = 10
    train_generate = partial(get_corpus_generate, filename='data/train.csv', batch_size=batch_size)
    test_generate = partial(get_corpus_generate, filename='data/test.csv', batch_size=batch_size)
    train_dataset = tf.data.Dataset.from_generator(train_generate,
                                                   output_types=(tf.int32, tf.float32),
                                                   output_shapes=(tf.TensorShape([7, batch_size, None]),
                                                                  tf.TensorShape([batch_size, class_num])))
    valid_dataset = tf.data.Dataset.from_generator(test_generate,
                                                   output_types=(tf.int32, tf.float32),
                                                   output_shapes=(
                                                       tf.TensorShape([7, batch_size, None]),
                                                       tf.TensorShape([batch_size, class_num]))
                                                   )
    model = RESEmbedding(class_num=class_num)
    model.build(input_shape=(7, 64, None))
    model.summary()
    train_model(model, train_dataset, valid_dataset, epoches, batch_size=batch_size)
    model.save_weights('./data/age.h5')
